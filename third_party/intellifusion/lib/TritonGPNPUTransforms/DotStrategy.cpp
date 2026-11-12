#include "Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/Sys/GetEnv.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <optional>

namespace mlir::triton::Intellifusion {
namespace {

constexpr StringLiteral kSkipAllMMAAttr = "ttg.skip_accelerate_matmul";
constexpr StringLiteral kSkipMMAv2Attr = "ttg.skip_mma_v2";
constexpr StringLiteral kStrategyAttr = "ttg.gpnpu.dot_strategy";
constexpr StringLiteral kTF32InputsAttr = "ttg.gpnpu.tf32_inputs";
constexpr StringLiteral kFMAStrategy = "fma";
constexpr StringLiteral kTF32FMAStrategy = "tf32-fma";

static bool isTF32Dot(DotOp dotOp) {
  Type aType = dotOp.getA().getType().getElementType();
  Type bType = dotOp.getB().getType().getElementType();
  return aType.isF32() && bType.isF32() &&
         dotOp.getInputPrecision() == InputPrecision::TF32;
}

// Describes how aggressively an MMAv5-capable dot should be padded to a
// native tile. Directly legal dots use MMAv5 for every policy except
// Unsupported.
enum class MMAv5PaddingPolicy {
  Unsupported,
  Optional,
  Preferred,
  Required,
};

struct MMAv5PaddingConfig {
  int64_t m;
  int64_t n;
  int64_t k;
};

static bool isFMAType(Type type) {
  return type.isF16() || type.isBF16() || type.isF32() || type.isF64();
}

static bool isFMAEligible(DotOp dotOp) {
  auto resultType = dotOp.getType();
  if (resultType.getRank() != 2 && resultType.getRank() != 3)
    return false;
  return isFMAType(dotOp.getA().getType().getElementType()) &&
         isFMAType(dotOp.getB().getType().getElementType()) &&
         isFMAType(resultType.getElementType());
}

static bool isInt8MMAv5Dot(DotOp dotOp) {
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  auto resultType = cast<RankedTensorType>(dotOp.getType());
  return aType.getElementType().isInteger(8) &&
         bType.getElementType().isInteger(8) &&
         resultType.getElementType().isInteger(32);
}

static bool isNativeFP8Type(Type type) {
  return llvm::isa<Float8E4M3FNType, Float8E5M2Type>(type);
}

static bool requiresMMAv5(DotOp dotOp) {
  if (isInt8MMAv5Dot(dotOp))
    return true;
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  return isNativeFP8Type(aType.getElementType()) &&
         isNativeFP8Type(bType.getElementType());
}

// This is the original GPNPU min_dot_size policy, moved from semantic.dot to
// the TTIR strategy gate. It deliberately does not include the full MMAv5
// legality check; AccelerateMatmul performs that check after this pass.
static bool meetsMMAv5MinSize(DotOp dotOp) {
  if (dotOp.getType().getRank() != 2)
    return false;

  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  if (aType.getRank() < 2 || bType.getRank() < 2)
    return false;

  int aBitwidth = aType.getElementType().getIntOrFloatBitWidth();
  int bBitwidth = bType.getElementType().getIntOrFloatBitWidth();
  if (aBitwidth != bBitwidth)
    return false;

  constexpr int64_t minM = 64;
  constexpr int64_t minN = 8;
  int64_t minK = aBitwidth == 8 ? 32 : 16;
  return aType.getShape()[aType.getRank() - 2] >= minM &&
         bType.getShape().back() >= minN && aType.getShape().back() >= minK;
}

static bool canPadDimension(int64_t original, int64_t padded) {
  return original > 0 && padded >= original && padded % original == 0 &&
         llvm::isPowerOf2_64(padded / original);
}

static constexpr std::optional<int64_t>
getPaddedDimension(int64_t original, int64_t minimum, int64_t alignment) {
  if (original <= 0)
    return std::nullopt;
  int64_t padded = original;
  while (padded < minimum || padded % alignment != 0) {
    if (padded > std::numeric_limits<int64_t>::max() / 2)
      return std::nullopt;
    padded *= 2;
  }
  return padded;
}

// A binary expansion must also reach the native MMAv5 tile alignment. Merely
// reaching the minimum would leave 48 -> 96 and 10 -> 20 MMA-ineligible.
static_assert(getPaddedDimension(48, 64, 64).value() == 192);
static_assert(getPaddedDimension(10, 8, 8).value() == 40);

// Return the tile to which a rank-2 dot can be zero-padded using the TTIR
// view/join primitives below. The current implementation deliberately limits
// this to power-of-two expansions, which covers tl.arange-based dot tiles and
// makes both padding and cropping an exact sequence of binary operations.
static std::optional<MMAv5PaddingConfig> getMMAv5PaddingConfig(DotOp dotOp) {
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  auto cType = cast<RankedTensorType>(dotOp.getC().getType());
  if (aType.getRank() != 2 || bType.getRank() != 2 || cType.getRank() != 2)
    return std::nullopt;

  const int64_t m = aType.getShape()[0];
  const int64_t k = aType.getShape()[1];
  const int64_t n = bType.getShape()[1];
  if (bType.getShape()[0] != k || cType.getShape()[0] != m ||
      cType.getShape()[1] != n)
    return std::nullopt;

  const int aBitwidth = aType.getElementType().getIntOrFloatBitWidth();
  const int bBitwidth = bType.getElementType().getIntOrFloatBitWidth();
  if (aBitwidth != bBitwidth)
    return std::nullopt;

  constexpr int64_t minM = 64;
  constexpr int64_t minN = 8;
  const int64_t minK = aBitwidth == 8 ? 32 : 16;
  auto paddedM = getPaddedDimension(m, minM, minM);
  // A native 8-bit tile narrower than 8 needs a 16-column padded tile:
  // its N-major B layout cannot use an 8-column MMAv5 instruction. An
  // explicit 8-column tile remains unchanged so the shared loader can
  // reject an illegal N-major layout or accept a legal K-major layout.
  const int64_t paddedNMin = requiresMMAv5(dotOp) && n < minN ? 16 : minN;
  auto paddedN = getPaddedDimension(n, paddedNMin, paddedNMin);
  auto paddedK = getPaddedDimension(k, minK, 1);
  if (!paddedM || !paddedN || !paddedK)
    return std::nullopt;
  MMAv5PaddingConfig config = {*paddedM, *paddedN, *paddedK};
  if (config.m == m && config.n == n && config.k == k)
    return std::nullopt;
  if (!canPadDimension(m, config.m) || !canPadDimension(n, config.n) ||
      !canPadDimension(k, config.k))
    return std::nullopt;
  return config;
}

static bool supportsMMAv5WarpCount(int numWarps) {
  return numWarps == 4 || numWarps == 8;
}

static bool isFP32Dot(DotOp dotOp) {
  return dotOp.getA().getType().getElementType().isF32() &&
         dotOp.getB().getType().getElementType().isF32() &&
         dotOp.getType().getElementType().isF32();
}

static std::optional<int64_t> getF32TensorCoreMinK(InputPrecision precision) {
  switch (precision) {
  case InputPrecision::TF32:
  case InputPrecision::TF32x3:
    return 8;
  case InputPrecision::BF16x3:
  case InputPrecision::BF16x6:
    return 16;
  case InputPrecision::IEEE:
    return std::nullopt;
  }
  llvm_unreachable("unknown dot input precision");
}

// F32 input precision selects a Tensor Core algorithm on GPNPU. Decomposed
// precision modes require MMAv5; only plain TF32 may fall back to TF32/FMA
// when the logical tile cannot use MMAv5. This check is stricter
// than meetsMMAv5MinSize because F32DotTC would otherwise erase the strategy
// marker before AccelerateMatmul discovers an alignment failure.
static bool isF32TensorCoreMMAv5Eligible(DotOp dotOp, int numWarps,
                                         int numCTAs) {
  if (!isFP32Dot(dotOp) || dotOp.getType().getRank() != 2 || numCTAs != 1 ||
      !supportsMMAv5WarpCount(numWarps) ||
      ::mlir::triton::tools::getBoolEnv("DISABLE_MMA_V5"))
    return false;

  auto minK = getF32TensorCoreMinK(dotOp.getInputPrecision());
  if (!minK)
    return false;

  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  if (aType.getRank() != 2 || bType.getRank() != 2)
    return false;

  const int64_t m = aType.getShape()[0];
  const int64_t k = aType.getShape()[1];
  const int64_t n = bType.getShape()[1];
  return m >= 64 && m % 64 == 0 && n >= 8 && n % 8 == 0 && k >= *minK;
}

static MMAv5PaddingPolicy getMMAv5PaddingPolicy(DotOp dotOp) {
  // Native FP8 and INT8 have no GPNPU FMA lowering, so an undersized dot must
  // be padded whenever the launch can use MMAv5.
  if (requiresMMAv5(dotOp))
    return MMAv5PaddingPolicy::Required;

  Type aType = dotOp.getA().getType().getElementType();
  Type bType = dotOp.getB().getType().getElementType();
  if (isa<Float16Type, BFloat16Type>(aType) &&
      isa<Float16Type, BFloat16Type>(bType))
    return MMAv5PaddingPolicy::Optional;
  // Precision-aware FP32 is handled before this generic padding path.
  return MMAv5PaddingPolicy::Unsupported;
}

// This is the performance-policy gate. Keep it separate from the mechanical
// padding transform: a future threshold model can enable selected shapes
// without changing the transformation or the MMAv5 lowering path.
static bool
isWithinAutomaticTF32PaddingLimit(DotOp dotOp,
                                  const MMAv5PaddingConfig &padding) {
  auto aType = cast<RankedTensorType>(dotOp.getA().getType());
  auto bType = cast<RankedTensorType>(dotOp.getB().getType());
  constexpr int64_t maxComputeExpansion = 4;
  int64_t remainingExpansion = maxComputeExpansion;
  auto consumeAxisExpansion = [&](int64_t original, int64_t padded) {
    int64_t axisExpansion = padded / original;
    if (axisExpansion > remainingExpansion)
      return false;
    remainingExpansion /= axisExpansion;
    return true;
  };
  return consumeAxisExpansion(aType.getShape()[0], padding.m) &&
         consumeAxisExpansion(bType.getShape()[1], padding.n) &&
         consumeAxisExpansion(aType.getShape()[1], padding.k);
}

static bool shouldPadToMMAv5(DotOp dotOp, const MMAv5PaddingConfig &padding,
                             MMAv5PaddingPolicy policy, bool forcePadding) {
  if (policy == MMAv5PaddingPolicy::Required)
    return true;
  if (policy == MMAv5PaddingPolicy::Preferred)
    return isWithinAutomaticTF32PaddingLimit(dotOp, padding);
  // TODO(gpnpu): Use measured FMA-vs-MMAv5 padding break-even thresholds by
  // dtype and (M, N, K) range. Keep this disabled until that data exists.
  return policy == MMAv5PaddingPolicy::Optional && forcePadding;
}

static SmallVector<int32_t> moveAxisToFrontOrder(unsigned rank, unsigned axis) {
  SmallVector<int32_t> order;
  order.reserve(rank);
  order.push_back(axis);
  for (unsigned i = 0; i != rank; ++i)
    if (i != axis)
      order.push_back(i);
  return order;
}

static SmallVector<int32_t> inversePermutation(ArrayRef<int32_t> order) {
  SmallVector<int32_t> inverse(order.size());
  for (auto [i, dim] : llvm::enumerate(order))
    inverse[dim] = i;
  return inverse;
}

static Value appendZeroAlongAxis(IRRewriter &rewriter, Value value,
                                 unsigned axis) {
  auto type = cast<RankedTensorType>(value.getType());
  assert(axis < type.getRank());
  auto loc = value.getLoc();
  auto zero = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(type.getElementType()));
  auto zeros = SplatOp::create(rewriter, loc, type, zero);

  SmallVector<int64_t> paddedShape(type.getShape());
  paddedShape[axis] *= 2;
  // Match tl.cat(value, zeros, dim=axis): join introduces a new minor factor,
  // then move that factor immediately before the concatenated axis and merge
  // the pair. This keeps the original tensor as the leading half of the axis.
  Value joined = JoinOp::create(rewriter, loc, value, zeros);
  SmallVector<int32_t> order;
  order.reserve(type.getRank() + 1);
  for (unsigned i = 0; i != axis; ++i)
    order.push_back(i);
  order.push_back(type.getRank());
  for (unsigned i = axis; i != type.getRank(); ++i)
    order.push_back(i);
  Value concatenated = TransOp::create(rewriter, loc, joined, order);
  return ReshapeOp::create(rewriter, loc, paddedShape, concatenated);
}

static Value padTensorToShape(IRRewriter &rewriter, Value value,
                              ArrayRef<int64_t> paddedShape) {
  auto type = cast<RankedTensorType>(value.getType());
  assert(type.getRank() == paddedShape.size());
  for (unsigned axis = 0; axis != paddedShape.size(); ++axis) {
    while (cast<RankedTensorType>(value.getType()).getShape()[axis] <
           paddedShape[axis])
      value = appendZeroAlongAxis(rewriter, value, axis);
  }
  return value;
}

// Keep the leading logical portion of an axis. SplitOp only removes a final
// dimension of size two, so reshape the padded axis to
// [2, ..., 2, logical-size], move each leading binary dimension to the end,
// and keep its lhs. This preserves the prefix rather than an interleaved
// subset of the padded result.
static Value cropLeadingAxis(IRRewriter &rewriter, Value value, unsigned axis,
                             int64_t logicalSize) {
  auto type = cast<RankedTensorType>(value.getType());
  assert(axis < type.getRank());
  assert(type.getShape()[axis] >= logicalSize);
  if (type.getShape()[axis] == logicalSize)
    return value;

  auto initialOrder = moveAxisToFrontOrder(type.getRank(), axis);
  value = TransOp::create(rewriter, value.getLoc(), value, initialOrder);
  auto movedType = cast<RankedTensorType>(value.getType());
  int64_t factor = movedType.getShape().front() / logicalSize;
  assert(llvm::isPowerOf2_64(factor));

  SmallVector<int64_t> splitShape;
  while (factor > 1) {
    splitShape.push_back(2);
    factor /= 2;
  }
  splitShape.push_back(logicalSize);
  llvm::append_range(splitShape, movedType.getShape().drop_front());
  value = ReshapeOp::create(rewriter, value.getLoc(), splitShape, value);

  while (cast<RankedTensorType>(value.getType()).getRank() > type.getRank()) {
    auto splitType = cast<RankedTensorType>(value.getType());
    SmallVector<int32_t> order;
    order.reserve(splitType.getRank());
    for (unsigned i = 1; i != splitType.getRank(); ++i)
      order.push_back(i);
    order.push_back(0);
    value = TransOp::create(rewriter, value.getLoc(), value, order);

    value = SplitOp::create(rewriter, value.getLoc(), value).getOutLHS();
  }
  return TransOp::create(rewriter, value.getLoc(), value,
                         inversePermutation(initialOrder));
}

static Value cropTensorToShape(IRRewriter &rewriter, Value value,
                               ArrayRef<int64_t> logicalShape) {
  auto type = cast<RankedTensorType>(value.getType());
  assert(type.getRank() == logicalShape.size());
  for (unsigned axis = 0; axis != logicalShape.size(); ++axis)
    value = cropLeadingAxis(rewriter, value, axis, logicalShape[axis]);
  return value;
}

static bool isZeroAccumulator(Value value) {
  return matchPattern(value, m_Zero());
}

static void padDotToMMAv5(IRRewriter &rewriter, DotOp dotOp,
                          const MMAv5PaddingConfig &padding,
                          MLIRContext *context) {
  auto resultType = cast<RankedTensorType>(dotOp.getType());
  SmallVector<int64_t> paddedAShape = {padding.m, padding.k};
  SmallVector<int64_t> paddedBShape = {padding.k, padding.n};
  SmallVector<int64_t> paddedResultShape = {padding.m, padding.n};

  auto paddedResultType =
      RankedTensorType::get(paddedResultShape, resultType.getElementType());

  Value paddedA = padTensorToShape(rewriter, dotOp.getA(), paddedAShape);
  Value paddedB = padTensorToShape(rewriter, dotOp.getB(), paddedBShape);
  const bool zeroAccumulator = isZeroAccumulator(dotOp.getC());
  Value paddedC;
  if (zeroAccumulator || requiresMMAv5(dotOp)) {
    auto zero = arith::ConstantOp::create(
        rewriter, dotOp.getLoc(),
        rewriter.getZeroAttr(resultType.getElementType()));
    paddedC = SplatOp::create(rewriter, dotOp.getLoc(), paddedResultType, zero);
  } else {
    paddedC = padTensorToShape(rewriter, dotOp.getC(), paddedResultShape);
  }

  auto paddedDot = DotOp::create(
      rewriter, dotOp.getLoc(), paddedResultType, paddedA, paddedB, paddedC,
      dotOp.getInputPrecision(), dotOp.getMaxNumImpreciseAcc());
  paddedDot->setAttr(kSkipMMAv2Attr, UnitAttr::get(context));
  paddedDot->setAttr(kStrategyAttr, StringAttr::get(context, "mmav5-padded"));
  if (dotOp->hasAttr(kTF32InputsAttr))
    paddedDot->setAttr(kTF32InputsAttr, UnitAttr::get(context));

  Value cropped = cropTensorToShape(rewriter, paddedDot, resultType.getShape());
  if (!zeroAccumulator && requiresMMAv5(dotOp)) {
    Type elementType = resultType.getElementType();
    if (isa<FloatType>(elementType))
      cropped = arith::AddFOp::create(rewriter, dotOp.getLoc(), cropped,
                                      dotOp.getC());
    else
      cropped = arith::AddIOp::create(rewriter, dotOp.getLoc(), cropped,
                                      dotOp.getC());
  }
  dotOp.replaceAllUsesWith(cropped);
  dotOp.erase();
}

class GPNPUDotStrategyPass
    : public PassWrapper<GPNPUDotStrategyPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPNPUDotStrategyPass)

  GPNPUDotStrategyPass() = default;
  explicit GPNPUDotStrategyPass(bool forcePadding = false, int numWarps = 4,
                                int numCTAs = 1) {
    this->forcePadding = forcePadding;
    this->numWarps = numWarps;
    this->numCTAs = numCTAs;
  }
  GPNPUDotStrategyPass(const GPNPUDotStrategyPass &pass) : PassWrapper(pass) {}

  StringRef getArgument() const final { return "gpnpu-dot-strategy"; }
  StringRef getDescription() const final {
    return "select GPNPU FMA or MMAv5 dot lowering";
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    bool failed = false;

    // IFWA disables MMAv2 fallback. Set the policy on the module so it
    // applies to tt.dot operations created by decomposing tt.dot_scaled.
    module->setAttr(kSkipMMAv2Attr, UnitAttr::get(context));

    SmallVector<DotOp> dots;
    module.walk([&](DotOp dotOp) { dots.push_back(dotOp); });

    const bool bMMAv5WarpCountSupported = supportsMMAv5WarpCount(numWarps);
    const bool mmav5Available =
        bMMAv5WarpCountSupported && numCTAs == 1 &&
        !::mlir::triton::tools::getBoolEnv("DISABLE_MMA_V5");
    IRRewriter rewriter(context);
    for (DotOp dotOp : dots) {
      if (isFP32Dot(dotOp) &&
          dotOp.getInputPrecision() != InputPrecision::IEEE) {
        if (isF32TensorCoreMMAv5Eligible(dotOp, numWarps, numCTAs)) {
          dotOp->setAttr(kSkipMMAv2Attr, UnitAttr::get(context));
          dotOp->setAttr(kStrategyAttr,
                         StringAttr::get(context, "f32-dot-tc-mmav5"));
        } else if (dotOp.getInputPrecision() != InputPrecision::TF32) {
          dotOp.emitError()
              << "cannot use MMAv5 for f32 dot with input_precision='"
              << stringifyInputPrecision(dotOp.getInputPrecision())
              << "': requires rank-2 operands, M >= 64 and divisible by 64, "
                 "N >= 8 and divisible by 8, K >= "
              << *getF32TensorCoreMinK(dotOp.getInputPrecision())
              << ", num_warps in {4, 8}, num_ctas = 1, and MMAv5 enabled";
          failed = true;
        } else {
          // Keep TF32 input semantics when MMAv5 is unavailable. The marker
          // tells the FMA lowering to truncate the original F32 operands.
          dotOp->setAttr(kTF32InputsAttr, UnitAttr::get(context));
          dotOp->setAttr(kSkipAllMMAAttr, UnitAttr::get(context));
          dotOp->setAttr(kStrategyAttr,
                         StringAttr::get(context, kTF32FMAStrategy));
        }
        continue;
      }

      MMAv5PaddingPolicy paddingPolicy = getMMAv5PaddingPolicy(dotOp);
      if (mmav5Available && paddingPolicy != MMAv5PaddingPolicy::Unsupported) {
        if (auto padding = getMMAv5PaddingConfig(dotOp);
            padding &&
            shouldPadToMMAv5(dotOp, *padding, paddingPolicy, forcePadding)) {
          rewriter.setInsertionPoint(dotOp);
          padDotToMMAv5(rewriter, dotOp, *padding, context);
          continue;
        }

        // An MMAv5-sized dot remains available to AccelerateMatmul, which
        // performs the full MMAv5 legality check.
        if (meetsMMAv5MinSize(dotOp)) {
          dotOp->setAttr(kSkipMMAv2Attr, UnitAttr::get(context));
          dotOp->setAttr(kStrategyAttr, StringAttr::get(context, "mmav5"));
          continue;
        }
      }

      // An FMA dot skips all MMA rewrites and is lowered by GPNPU's existing
      // DotOpToLLVM path.
      if (isFMAEligible(dotOp)) {
        dotOp->setAttr(kSkipAllMMAAttr, UnitAttr::get(context));
        dotOp->setAttr(kStrategyAttr,
                       StringAttr::get(context, isTF32Dot(dotOp)
                                                    ? kTF32FMAStrategy
                                                    : kFMAStrategy));
        continue;
      }

      auto diagnostic = dotOp.emitError();
      diagnostic << "GPNPU dot is neither directly legal for MMAv5 nor "
                    "supported by the FMA fallback";
      if (requiresMMAv5(dotOp) && dotOp.getType().getRank() == 2) {
        auto aType = cast<RankedTensorType>(dotOp.getA().getType());
        auto bType = cast<RankedTensorType>(dotOp.getB().getType());
        int64_t m = aType.getShape()[0];
        int64_t k = aType.getShape()[1];
        int64_t n = bType.getShape()[1];
        diagnostic << "; rank-2 INT8/FP8 MMAv5 tile requirements: "
                      "BLOCK_M >= 64 and divisible by 64, BLOCK_N >= 8 and "
                      "divisible by 8, BLOCK_K >= 32, num_warps in {4, 8}, "
                      "and MMAv5 enabled; current tile (M="
                   << m << ", N=" << n << ", K=" << k
                   << "), num_warps=" << static_cast<int>(numWarps)
                   << "; for N-major native 8-bit B use BLOCK_N >= 16 "
                      "and divisible by 16; try BLOCK_M=64, BLOCK_N=16, "
                      "BLOCK_K=32, num_warps=4 (with masks for smaller "
                      "logical tensors)";
      }
      failed = true;
    }

    if (failed)
      signalPassFailure();
  }

private:
  Option<bool> forcePadding{*this, "force-padding",
                            llvm::cl::desc("force eligible MMAv5 padding"),
                            llvm::cl::init(false)};
  Option<int> numWarps{*this, "num-warps",
                       llvm::cl::desc("number of GPNPU warps"),
                       llvm::cl::init(4)};
  Option<int> numCTAs{*this, "num-ctas", llvm::cl::desc("number of GPNPU CTAs"),
                      llvm::cl::init(1)};
};

// Triton 3.8 can feed a distributed scale tensor directly into tmem_alloc.
// The upstream convert-layout lowering is valid for block16 MMA,
// but GPNPU's scale_vec::4X instruction requires the pre-3.8 shared-memory
// staging layout. Recreate that layout before shared-memory allocation so the
// regular allocation and lowering passes account for it normally.
static void stageNvFp4Scale(IRRewriter &rewriter, Value scale) {
  auto tmemAlloc = scale.getDefiningOp<nvidia_gpu::TMEMAllocOp>();
  if (!tmemAlloc || !tmemAlloc.getSrc())
    return;

  auto convert = tmemAlloc.getSrc().getDefiningOp<gpu::ConvertLayoutOp>();
  if (!convert)
    return;

  Value source = convert.getSrc();
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  auto stagedType = dyn_cast<RankedTensorType>(convert.getType());
  if (!sourceType || !stagedType)
    return;

  MLIRContext *context = rewriter.getContext();
  auto cgaLayout = gpu::getCGALayout(sourceType.getEncoding());
  auto sharedLayout = gpu::NVMMASharedEncodingAttr::get(
      context, /*swizzlingByteWidth=*/0, /*transposed=*/false,
      sourceType.getElementType().getIntOrFloatBitWidth(),
      /*fp4Padded=*/false, cgaLayout);
  auto sharedType = gpu::MemDescType::get(
      sourceType.getShape(), sourceType.getElementType(), sharedLayout,
      gpu::SharedMemorySpaceAttr::get(context));

  rewriter.setInsertionPoint(convert);
  auto localAlloc =
      gpu::LocalAllocOp::create(rewriter, convert.getLoc(), sharedType, source);
  auto localLoad = gpu::LocalLoadOp::create(rewriter, convert.getLoc(),
                                            stagedType, localAlloc.getResult());
  rewriter.modifyOpInPlace(tmemAlloc, [&] {
    tmemAlloc.getSrcMutable().assign(localLoad.getResult());
  });
  if (convert->use_empty())
    rewriter.eraseOp(convert);
}

class GPNPUStageScaledFp4Pass
    : public PassWrapper<GPNPUStageScaledFp4Pass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPNPUStageScaledFp4Pass)

  StringRef getArgument() const final { return "gpnpu-stage-scaled-fp4"; }
  StringRef getDescription() const final {
    return "stage NVFP4 MMAv5 scales through GPNPU-compatible shared memory";
  }

  void runOnOperation() final {
    SmallVector<nvidia_gpu::TCGen5MMAScaledOp> scaledMmas;
    getOperation().walk([&](nvidia_gpu::TCGen5MMAScaledOp op) {
      auto aScaleType = cast<gpu::MemDescType>(op.getAScale().getType());
      auto bScaleType = cast<gpu::MemDescType>(op.getBScale().getType());
      if (op.getAType() == ScaleDotElemType::E2M1 &&
          op.getBType() == ScaleDotElemType::E2M1 &&
          isa<Float8E4M3FNType>(aScaleType.getElementType()) &&
          isa<Float8E4M3FNType>(bScaleType.getElementType()))
        scaledMmas.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (auto op : scaledMmas) {
      stageNvFp4Scale(rewriter, op.getAScale());
      stageNvFp4Scale(rewriter, op.getBScale());
    }
  }
};

} // namespace

std::unique_ptr<Pass> createGPNPUDotStrategyPass(bool forcePadding,
                                                 int numWarps, int numCTAs) {
  return std::make_unique<GPNPUDotStrategyPass>(forcePadding, numWarps,
                                                numCTAs);
}

std::unique_ptr<Pass> createGPNPUStageScaledFp4Pass() {
  return std::make_unique<GPNPUStageScaledFp4Pass>();
}

} // namespace mlir::triton::Intellifusion
