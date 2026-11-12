#include "Passes.h"
#include "intellifusion/lib/TargetCapabilities.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/DecomposeScaledBlocked.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace triton {
namespace Intellifusion {

using namespace mlir::triton::gpu;

namespace {

static bool supportsGPNPUMMAv5(DotOp op) {
  if (!target::kSupportsMMAv5 || !supportMMA(op, /*version=*/5)) {
    auto remark = op.emitRemark()
                  << "IFWA MMA version 5 acceleration was not applied due to "
                     "unsupported shapes or data types";
    remark.attachNote()
        << "IFWA supports MMA version 5 but intentionally provides no "
           "MMA v2/v3 fallback";
    return false;
  }
  return true;
}

static bool skipAllMMA(DotOp dotOp) {
  return dotOp->hasAttr("ttg.skip_accelerate_matmul");
}

// Returns a shared memory allocation that can be used by a dotMMA op for the
// given value.
static Value
getSharedMemoryMMAOperand(Value v, mlir::PatternRewriter &rewriter, int opIdx,
                          bool allowTranspose, bool isMMAv5Fp4Padded = false,
                          bool forceTranspose = false,
                          Operation *op = nullptr /*only for diagnostic*/) {
  OpBuilder::InsertionGuard g(rewriter);
  Value arg = v;
  while (auto cvtOp = arg.getDefiningOp<ConvertLayoutOp>())
    arg = cvtOp.getSrc();
  auto argType = cast<RankedTensorType>(arg.getType());
  assert(argType.getEncoding() && "unexpected tensor type");
  auto order = getOrderForMemory(argType);

  // If the MMA op doesn't support transpose pick the layout expected by the MMA
  // op.
  llvm::SmallVector<unsigned> newOrder = order;
  if (!allowTranspose) {
    if (opIdx == 1) {
      newOrder = {0, 1};
    } else {
      newOrder = {1, 0};
    }
    if (forceTranspose)
      std::swap(newOrder[0], newOrder[1]);
  }

  if (newOrder != order && op) {
    op->emitWarning("Warning: Forcing a different order [")
        << newOrder[0] << ", " << newOrder[1]
        << "] on SMEM than the register order for the operand " << opIdx
        << ". Registers will be transposed before SMEM store and the pipelined "
           "load for this operand will be disabled, so poor performance is "
           "expected. Recommendation: consider transposing the operand in "
           "global "
           "memory to remove the need to transpose the tensor in registers.";
  }

  Attribute SharedMemorySpace =
      SharedMemorySpaceAttr::get(argType.getContext());
  auto CGALayout = getCGALayout(argType.getEncoding());
  auto newLayout = NVMMASharedEncodingAttr::get(
      argType.getContext(), argType.getShape(), newOrder, CGALayout,
      argType.getElementType(), isMMAv5Fp4Padded);
  auto newType = MemDescType::get(argType.getShape(), argType.getElementType(),
                                  newLayout, SharedMemorySpace);
  rewriter.setInsertionPointAfterValue(arg);
  return LocalAllocOp::create(rewriter, arg.getLoc(), newType, arg);
}

static bool isTmemCopyCompatibleScale(RankedTensorType argType,
                                      bool usesTMAload) {
  assert(argType.getEncoding() && "unexpected tensor type");
  {
    auto shmemLayout = NVMMASharedEncodingAttr::get(
        argType.getContext(), /*swizzlingByteWidth=*/0,
        /*transposed=*/false,
        /*elementBitWidth=*/8,
        /*fp4Padded=*/false, getCGALayout(argType.getEncoding()));
    auto type = MemDescType::get(
        argType.getShape(), argType.getElementType(), shmemLayout,
        SharedMemorySpaceAttr::get(argType.getContext()));
    if (!isInnermostContiguous(type, 512)) {
      // TMEM copy expects metadata "chunks" in SMEM to be stored contiguously
      // in the innermost axes.
      return false;
    }
  }

  if (usesTMAload)
    return true;

  if (argType.getRank() != 2) {
    // TODO: Add support for higher rank when 5D coalesced load is fixed.
    return false;
  }

  auto innermostBits = argType.getDimSize(argType.getRank() - 1) * 8;
  return innermostBits % (32 * 128) == 0;
}

template <typename OpTy>
static OpTy getDefiningOpSkippingConvertLayout(Value value) {
  while (auto cvtOp = value.getDefiningOp<ConvertLayoutOp>())
    value = cvtOp.getSrc();
  return value.getDefiningOp<OpTy>();
}

static Value stripConvertLayout(Value value) {
  while (auto cvtOp = value.getDefiningOp<ConvertLayoutOp>())
    value = cvtOp.getSrc();
  return value;
}

static Value
tryCreateTmemCopyCompatibleScaleOperand(Value scale,
                                        mlir::PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  // The final reshape to conform to the logical shape requirement of
  // dot_scaled.
  auto reshape2D = getDefiningOpSkippingConvertLayout<ReshapeOp>(scale);
  if (!reshape2D)
    return Value();

  auto scaleType = dyn_cast<RankedTensorType>(reshape2D.getResult().getType());
  if (!scaleType || scaleType.getShape().size() != 2)
    return Value();

  auto scale2DShape = scaleType.getShape();
  auto blockMN = scale2DShape[0];
  auto numScales = scale2DShape[1];
  if (blockMN % 128 != 0 || numScales % 4 != 0)
    return Value();

  // Permute to expose the 32x4x4 interleaving used by scale blocks in TMEM.
  auto transOp =
      getDefiningOpSkippingConvertLayout<TransOp>(reshape2D.getSrc());
  if (!transOp || transOp.getOrder() != ArrayRef<int32_t>({0, 3, 2, 1, 4}))
    return Value();

  // A logical scale tensor has shape [block_mn, num_scales]. The tensor view
  // chain in a user kernel signals TMEMCopy compatibility by exposing the
  // underlying 128x4 scale block as
  //     --reshape
  //   [mn_copy_tiles, k_copy_tiles, 32, 4, 4]
  //     --trans {0, 3, 2, 1, 4}-->
  //   [mn_copy_tiles, 4, 32, k_copy_tiles, 4]
  //     --flatten-->
  //   [block_mn, num_scales]
  //
  // The block shape with which the scale tensor is loaded is flexible:
  // For example, [.., 2, 256], [.., 512], and [.., 32, 4, 4] are all acceptable
  // as long as a reshape exposes the same 32x4x4 blocks before the transpose.
  auto mnCopyTiles = blockMN / 128;
  auto kCopyTiles = numScales / 4;
  SmallVector<int64_t> tiledScaleShape = {mnCopyTiles, kCopyTiles, 32, 4, 4};

  Value tiledScale = stripConvertLayout(transOp.getSrc());
  auto reshapeTiled = tiledScale.getDefiningOp<ReshapeOp>();
  if (!reshapeTiled ||
      reshapeTiled.getType().getShape() != ArrayRef<int64_t>(tiledScaleShape))
    return Value();

  Value packedScale = stripConvertLayout(reshapeTiled.getSrc());
  auto packedScaleType = dyn_cast<RankedTensorType>(packedScale.getType());
  if (!packedScaleType)
    return Value();

  auto *loadOp = packedScale.getDefiningOp();
  if (!isa_and_nonnull<LoadOp, DescriptorLoadLikeOpInterface>(loadOp))
    return Value();

  bool usesTMAload = isa<DescriptorLoadLikeOpInterface>(loadOp);
  if (!isTmemCopyCompatibleScale(packedScaleType, usesTMAload))
    return Value();

  auto sharedMemory = SharedMemorySpaceAttr::get(scaleType.getContext());
  auto sinkLayout = getScaleSmemLayoutForTMEMCopy(
      scaleType.getContext(), scaleType.getShape(),
      getCGALayout(scaleType.getEncoding()));
  auto layout = SharedLinearEncodingAttr::get(scaleType.getContext(),
                                              std::move(sinkLayout),
                                              /*alignment=*/128);
  auto sinkType =
      MemDescType::get(scaleType.getShape(), scaleType.getElementType(), layout,
                       sharedMemory, /*mutableMemory=*/false);
  rewriter.setInsertionPointAfterValue(reshape2D.getResult());
  return LocalAllocOp::create(rewriter, reshape2D.getLoc(), sinkType,
                              reshape2D.getResult());
}

static Value createTmemScaleOperand(Value tensor, Attribute tmemEncoding,
                                    Attribute tensorMemorySpace, int numWarps,
                                    Location loc, PatternRewriter &rewriter) {
  auto tensorType = cast<RankedTensorType>(tensor.getType());
  auto tmemType =
      MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                       tmemEncoding, tensorMemorySpace,
                       /*mutableMemory=*/false);
  auto tmemLayout = nvidia_gpu::getDefaultLayoutForTmemLdSt(tmemType, numWarps);
  auto stagedType = tensorType.cloneWithEncoding(tmemLayout);
  Value converted = ConvertLayoutOp::create(rewriter, loc, stagedType, tensor);
  return triton::nvidia_gpu::TMEMAllocOp::create(rewriter, loc, tmemType,
                                                 /*token=*/Type(), converted);
}

static bool bwdFilter(Operation *op) {
  return (op->hasTrait<OpTrait::Elementwise>() && isMemoryEffectFree(op)) ||
         isView(op) ||
         isa<Fp4ToFpOp, LoadOp, DescriptorLoadLikeOpInterface, BroadcastOp,
             ConvertLayoutOp>(op);
}

static bool joinPacksLastDim(JoinOp join) {
  auto type = cast<RankedTensorType>(join.getResult().getType());
  unsigned rank = type.getRank();
  if (rank < 2)
    return false;

  auto order = getOrder(type);
  // Check that the appended join axis and the previous last axis are the two
  // fastest-changing axes, so the join packs the last dimension contiguously.
  return order[0] == rank - 1 && order[1] == rank - 2;
}

static bool canWidenKWidthForJoin(int loadBitWidth,
                                  const SetVector<Operation *> &slice) {
  return llvm::any_of(slice, [loadBitWidth](Operation *op) {
    auto join = dyn_cast<JoinOp>(op);
    if (!join)
      return false;
    auto type = cast<RankedTensorType>(join.getResult().getType());
    return type.getElementTypeBitWidth() == loadBitWidth * 2 &&
           joinPacksLastDim(join);
  });
}

// Finds the bitwidth with which the value x is loaded
static int computeOrigBitWidth(Value x) {
  SetVector<Operation *> slice;
  mlir::BackwardSliceOptions opt;
  opt.omitBlockArguments = true;
  opt.filter = bwdFilter;
  (void)getBackwardSlice(x, &slice, opt);

  // TODO: This heuristic may be a bit too coarse and may need improving
  // If the chain contains a fp4 to fp16/bf16 conversion, then the original
  // bitwidth is 4.
  if (llvm::any_of(slice, [](Operation *op) { return isa<Fp4ToFpOp>(op); }))
    return 4;

  int origBitWidth = getElementTypeOrSelf(x).getIntOrFloatBitWidth();
  for (auto op : slice) {
    if (isa<LoadOp, DescriptorLoadLikeOpInterface>(op)) {
      if (auto tensorTy =
              dyn_cast<RankedTensorType>(op->getResultTypes().front())) {
        origBitWidth =
            std::min<int>(origBitWidth, tensorTy.getElementTypeBitWidth());
      }
    }
  }

  // If JoinOp occurred at least once, in backward layout propagation,
  // the kWidth will be split in half as we pass through the JoinOp.
  // Hence we divide origBitWidth by 2 here to compensate for that and
  // improve our load width.
  // This won't be optimal if there is a tree of multiple JoinOps, which
  // would require counting the max number of JoinOp's along any path.
  //
  // In the future we might want to do something like trying a large kWidth,
  // run layout backpropagation and see what's the contiguity that you
  // get at the loads that feed into it.
  //
  // This heuristic is intended for packed-K values of shape [..., K / 2]
  // joined into [..., K / 2, 2]. We also check that the bitwidth of the joined
  // value is twice the original bitwidth, meaning "one loaded storage element
  // contributes two logical K values after unpack/join".
  if (canWidenKWidthForJoin(origBitWidth, slice))
    origBitWidth /= 2;

  return origBitWidth;
}

class BlockedToMMAv5 : public mlir::OpRewritePattern<DotOp> {
public:
  BlockedToMMAv5(mlir::MLIRContext *context, int benefit)
      : OpRewritePattern<DotOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (skipAllMMA(dotOp))
      return failure();

    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = lookupNumWarps(dotOp);
    auto CGALayout = getCGALayout(oldRetType.getEncoding());

    if (!supportsGPNPUMMAv5(dotOp))
      return failure();
    constexpr int versionMajor = 5;
    Location loc = dotOp.getLoc();
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();
    if (std::min(computeOrigBitWidth(a), computeOrigBitWidth(b)) >= 32 &&
        dotOp.getInputPrecision() != InputPrecision::TF32)
      return failure();
    auto oldAType = dotOp.getA().getType();
    // TF32 transpose is only supported with 128 swizzle mode with 32B
    // atomicity. As we currently don't support this layout we disallow
    // transpose for TF32 inputs.
    bool allowTranspose = !dotOp.getA().getType().getElementType().isF32();
    a = getSharedMemoryMMAOperand(a, rewriter, 0, allowTranspose);
    b = getSharedMemoryMMAOperand(b, rewriter, 1, allowTranspose);
    MLIRContext *context = dotOp->getContext();
    auto instrShape = mmaVersionToInstrShape(
        versionMajor, retShapePerCTA, oldAType.getElementType(), numWarps);
    auto bitwidth = oldRetType.getElementType().getIntOrFloatBitWidth();
    unsigned colStride = 32 / bitwidth;
    Attribute accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
        context, instrShape[0], instrShape[1], colStride, CGALayout,
        /*twoCTAs=*/false);
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    MemDescType accMemDescType =
        MemDescType::get(oldRetType.getShape(), oldRetType.getElementType(),
                         accEncoding, tensorMemorySpace,
                         /*mutableMemory=*/true);
    auto newDistributedEncoding =
        nvidia_gpu::getDefaultLayoutForTmemLdSt(accMemDescType, numWarps);
    auto newAccType = oldRetType.cloneWithEncoding(newDistributedEncoding);
    Value cvtAcc =
        ConvertLayoutOp::create(rewriter, loc, newAccType, dotOp.getOperand(2));
    auto tokType = rewriter.getType<AsyncTokenType>();
    auto acc = triton::nvidia_gpu::TMEMAllocOp::create(
        rewriter, loc, accMemDescType, tokType, cvtAcc);
    auto vTrue = arith::ConstantIntOp::create(rewriter, dotOp.getLoc(), 1, 1);
    auto mma = triton::nvidia_gpu::TCGen5MMAOp::create(
        rewriter, loc, tokType, a, b, acc, acc.getToken(), /*useD=*/vTrue,
        /*pred=*/vTrue, /*two_ctas=*/false);

    auto ld = triton::nvidia_gpu::TMEMLoadOp::create(
        rewriter, loc, newAccType, tokType, acc, /*dep=*/mma.getToken());
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType, ld);
    return success();
  }
};

// Keep batch pairs in one thread so tt.split can
// decompose rank-3 scaled dots even when the matrix dimensions are small.
class SplitBatchedScaledDot
    : public mlir::OpRewritePattern<triton::DotScaledOp> {
public:
  SplitBatchedScaledDot(mlir::MLIRContext *context, int benefit)
      : mlir::OpRewritePattern<triton::DotScaledOp>(context, benefit) {}

  LogicalResult matchAndRewrite(triton::DotScaledOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto retTy = dotOp.getType();
    if (retTy.getRank() != 3 || !retTy.getEncoding() ||
        !isa<BlockedEncodingAttr>(retTy.getEncoding()))
      return failure();

    int64_t batch = retTy.getDimSize(0);
    if (!llvm::isPowerOf2_64(batch))
      return failure();

    Location loc = dotOp.getLoc();
    auto module = dotOp->getParentOfType<ModuleOp>();
    int numWarps = lookupNumWarps(dotOp);
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(module);
    int numCTAs = TritonGPUDialect::getNumCTAs(module);
    auto split = [&](Value value) -> SmallVector<Value> {
      auto ty = cast<RankedTensorType>(value.getType());
      Value transposed =
          TransOp::create(rewriter, loc, value, SmallVector<int32_t>{1, 2, 0});
      SmallVector<int64_t> shape = {ty.getDimSize(1), ty.getDimSize(2)};
      for (int64_t size = batch; size > 1; size /= 2)
        shape.push_back(2);
      Value reshaped = ReshapeOp::create(rewriter, loc, shape, transposed);
      // tt.split requires each batch pair to live in one thread. Default
      // layouts may distribute the pair across warps for small matrices.
      auto matrixEnc = getDefaultBlockedEncoding(
          rewriter.getContext(), ArrayRef<int64_t>(shape).take_front(2),
          numWarps, threadsPerWarp, numCTAs);
      SmallVector<unsigned> sizePerThread(matrixEnc.getSizePerThread());
      SmallVector<unsigned> threadsPerWarpVec(matrixEnc.getThreadsPerWarp());
      SmallVector<unsigned> warpsPerCTAVec(matrixEnc.getWarpsPerCTA());
      SmallVector<unsigned> order(matrixEnc.getOrder());
      for (unsigned dim = 2; dim < shape.size(); ++dim) {
        sizePerThread.push_back(2);
        threadsPerWarpVec.push_back(1);
        warpsPerCTAVec.push_back(1);
        order.insert(order.begin(), dim);
      }
      auto fullEnc = getDefaultBlockedEncoding(
          rewriter.getContext(), shape, numWarps, threadsPerWarp, numCTAs);
      auto splitEnc = BlockedEncodingAttr::get(
          rewriter.getContext(), sizePerThread, threadsPerWarpVec,
          warpsPerCTAVec, order, getCGALayout(fullEnc));
      auto reshapedTy = cast<RankedTensorType>(reshaped.getType());
      Value coalesced = ConvertLayoutOp::create(
          rewriter, loc, reshapedTy.cloneWithEncoding(splitEnc), reshaped);
      SmallVector<Value> parts = {coalesced};
      for (int64_t size = batch; size > 1; size /= 2) {
        SmallVector<Value> next;
        for (Value part : parts) {
          auto halves = SplitOp::create(rewriter, loc, part);
          next.push_back(halves.getOutLHS());
          next.push_back(halves.getOutRHS());
        }
        parts = std::move(next);
      }
      return parts;
    };

    auto a = split(dotOp.getA());
    auto b = split(dotOp.getB());
    auto c = split(dotOp.getC());
    auto aScale = dotOp.getAScale() ? split(dotOp.getAScale())
                                    : SmallVector<Value>(batch);
    auto bScale = dotOp.getBScale() ? split(dotOp.getBScale())
                                    : SmallVector<Value>(batch);

    auto toBlocked = [&](Value value) -> Value {
      auto ty = cast<RankedTensorType>(value.getType());
      auto enc = getDefaultBlockedEncoding(rewriter.getContext(), ty.getShape(),
                                           numWarps, threadsPerWarp, numCTAs);
      return ConvertLayoutOp::create(rewriter, loc, ty.cloneWithEncoding(enc),
                                     value);
    };

    SmallVector<Value> results;
    for (int64_t i = 0; i < batch; ++i) {
      Value ai = toBlocked(a[i]);
      Value bi = toBlocked(b[i]);
      Value ci = toBlocked(c[i]);
      Value sai = aScale[i] ? toBlocked(aScale[i]) : Value();
      Value sbi = bScale[i] ? toBlocked(bScale[i]) : Value();
      results.push_back(DotScaledOp::create(
          rewriter, loc, ci.getType(), ai, bi, ci, sai, sbi,
          dotOp.getAElemType(), dotOp.getBElemType(), dotOp.getFastMath(),
          dotOp.getLhsKPack(), dotOp.getRhsKPack()));
    }

    while (results.size() > 1) {
      SmallVector<Value> joined;
      for (size_t i = 0; i < results.size(); i += 2)
        joined.push_back(
            JoinOp::create(rewriter, loc, results[i], results[i + 1]));
      results = std::move(joined);
    }
    Value reshaped = ReshapeOp::create(
        rewriter, loc,
        SmallVector<int64_t>{retTy.getDimSize(1), retTy.getDimSize(2), batch},
        results.front());
    Value transposed =
        TransOp::create(rewriter, loc, reshaped, SmallVector<int32_t>{2, 0, 1});
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, retTy, transposed);
    return success();
  }
};

class ScaledBlockedToMMAv5
    : public mlir::OpRewritePattern<triton::DotScaledOp> {
public:
  ScaledBlockedToMMAv5(mlir::MLIRContext *context, int benefit)
      : mlir::OpRewritePattern<triton::DotScaledOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(triton::DotScaledOp dotOp,
                  mlir::PatternRewriter &rewriter) const override {
    RankedTensorType oldRetType = dotOp.getType();
    if (!oldRetType.getEncoding() ||
        mlir::isa<NvidiaMmaEncodingAttr>(oldRetType.getEncoding()))
      return failure();

    if (dotOp.getAScale() == nullptr || dotOp.getBScale() == nullptr) {
      return failure();
    }

    // get MMA encoding for the given number of warps
    auto retShapePerCTA = getShapePerCTA(oldRetType);
    int numWarps = lookupNumWarps(dotOp);
    auto CGALayout = getCGALayout(oldRetType.getEncoding());
    if (!target::kSupportsMMAv5)
      return failure();
    if (numWarps != 4 && numWarps != 8)
      return failure();
    if (retShapePerCTA[0] < 128 || retShapePerCTA[1] < 16)
      return failure();
    Location loc = dotOp.getLoc();
    // operands
    Value a = dotOp.getA();
    Value b = dotOp.getB();

    bool IsAMixedPrecFp4 = false;
    bool IsBMixedPrecFp4 = false;
    bool isAFP4 = dotOp.getAElemType() == ScaleDotElemType::E2M1;
    bool isBFP4 = dotOp.getBElemType() == ScaleDotElemType::E2M1;

    if (dotOp.getAElemType() != dotOp.getBElemType()) {
      if (isAFP4)
        IsAMixedPrecFp4 = true;
      else if (isBFP4)
        IsBMixedPrecFp4 = true;
    }

    bool isFp4MMA = isAFP4 && isBFP4;
    constexpr bool requiresFp4Padding = target::kRequiresFp4Padding;

    // IFWA requires shared-memory padding for the FP4 operand of
    // mixed-precision MMA. For FP4 x FP4, pad M-major A and N-major B.
    auto blockK = dotOp.getA().getType().getShape().back() * (isAFP4 ? 2 : 1);
    auto blockM = dotOp.getA().getType().getShape()[0];
    bool isMMAv5Fp4PaddedLhs =
        (isFp4MMA && !dotOp.getLhsKPack()) || // fp4  x fp4 with M-major A
        (IsAMixedPrecFp4 && (requiresFp4Padding || blockM == 64 ||
                             blockK == 32 || !dotOp.getLhsKPack()));
    bool isMMAv5Fp4PaddedRhs =
        (isFp4MMA && !dotOp.getRhsKPack()) || // fp4  x fp4 with N-major B
        (IsBMixedPrecFp4 &&
         (requiresFp4Padding || blockK == 32 || !dotOp.getRhsKPack()));

    // For mixed-precision fp4 operands, set allowTranspose = false, to force
    // the packed axis, K, to be contiguous in SMEM
    a = getSharedMemoryMMAOperand(a, rewriter, 0,
                                  /*allowTranspose=*/!isAFP4,
                                  /*isMMAv5Fp4Padded=*/isMMAv5Fp4PaddedLhs,
                                  /*forceTranspose=*/!dotOp.getLhsKPack(),
                                  dotOp);
    b = getSharedMemoryMMAOperand(b, rewriter, 1,
                                  /*allowTranspose=*/!isBFP4,
                                  /*isMMAv5Fp4Padded=*/isMMAv5Fp4PaddedRhs,
                                  /*forceTranspose=*/!dotOp.getRhsKPack(),
                                  dotOp);

    MLIRContext *context = dotOp->getContext();
    unsigned m = 128;
    unsigned n = retShapePerCTA[1] >= 256 ? 256 : retShapePerCTA[1];

    auto bitwidth = oldRetType.getElementType().getIntOrFloatBitWidth();
    unsigned colStride = 32 / bitwidth;
    Attribute accEncoding = triton::nvidia_gpu::TensorMemoryEncodingAttr::get(
        context, m, n, colStride, CGALayout);
    Attribute tensorMemorySpace =
        triton::nvidia_gpu::TensorMemorySpaceAttr::get(context);
    MemDescType accMemDescType =
        MemDescType::get(oldRetType.getShape(), oldRetType.getElementType(),
                         accEncoding, tensorMemorySpace,
                         /*mutableMemory=*/true);
    auto newDistributedEncoding =
        nvidia_gpu::getDefaultLayoutForTmemLdSt(accMemDescType, numWarps);
    auto newAccType = oldRetType.cloneWithEncoding(newDistributedEncoding);
    Value cvtAcc =
        ConvertLayoutOp::create(rewriter, loc, newAccType, dotOp.getOperand(2));
    auto tokType = rewriter.getType<AsyncTokenType>();
    auto acc = triton::nvidia_gpu::TMEMAllocOp::create(
        rewriter, loc, accMemDescType, tokType, cvtAcc);

    RankedTensorType oldScaleAType = dotOp.getAScale().getType();
    RankedTensorType oldScaleBType = dotOp.getBScale().getType();

    auto aScaleBlockRepOrder =
        triton::nvidia_gpu::getTensorMemoryScalesBlockRepOrder(
            dotOp, /*isA=*/true, dotOp.getAElemType(), dotOp.getBElemType(),
            oldScaleAType.getElementType(), oldScaleBType.getElementType());
    auto bScaleBlockRepOrder =
        triton::nvidia_gpu::getTensorMemoryScalesBlockRepOrder(
            dotOp, /*isA=*/false, dotOp.getAElemType(), dotOp.getBElemType(),
            oldScaleAType.getElementType(), oldScaleBType.getElementType());
    Attribute scaleAEncoding =
        triton::nvidia_gpu::TensorMemoryScalesEncodingAttr::get(
            context, CGALayout, aScaleBlockRepOrder);
    Attribute scaleBEncoding =
        triton::nvidia_gpu::TensorMemoryScalesEncodingAttr::get(
            context, CGALayout, bScaleBlockRepOrder);
    Value scaleA =
        tryCreateTmemCopyCompatibleScaleOperand(dotOp.getAScale(), rewriter);
    if (!scaleA) {
      scaleA =
          createTmemScaleOperand(dotOp.getAScale(), scaleAEncoding,
                                 tensorMemorySpace, numWarps, loc, rewriter);
    }
    Value scaleB =
        tryCreateTmemCopyCompatibleScaleOperand(dotOp.getBScale(), rewriter);
    if (!scaleB) {
      scaleB =
          createTmemScaleOperand(dotOp.getBScale(), scaleBEncoding,
                                 tensorMemorySpace, numWarps, loc, rewriter);
    }

    auto vTrue = arith::ConstantIntOp::create(rewriter, dotOp.getLoc(), 1, 1);
    auto mmaOp = triton::nvidia_gpu::TCGen5MMAScaledOp::create(
        rewriter, loc, tokType, a, b, acc.getResult(), acc.getToken(), scaleA,
        scaleB, dotOp.getAElemType(), dotOp.getBElemType(),
        /*useD=*/vTrue, /*pred=*/vTrue);

    auto ld = triton::nvidia_gpu::TMEMLoadOp::create(
        rewriter, loc, newAccType, tokType, acc, mmaOp.getToken());
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(dotOp, oldRetType, ld);
    return success();
  }
};
} // namespace

static Value promoteOperand(OpBuilder &builder, Location loc, Value operand,
                            Type promotedType) {
  Type tensorPromotedType = cast<RankedTensorType>(operand.getType())
                                .cloneWith(std::nullopt, promotedType);
  Type operandElType =
      cast<RankedTensorType>(operand.getType()).getElementType();
  if (type::isFloat8(operandElType)) {
    return FpToFpOp::create(builder, loc, tensorPromotedType, operand);
  }
  return arith::ExtFOp::create(builder, loc, tensorPromotedType, operand);
}

// promote operands of dot op if the existing combination is not natively
// supported.
static LogicalResult decomposeMixedModeDotOp(ModuleOp mod) {
  bool foundUnsupportedMMA = false;
  mod.walk([&](DotOp dotOp) -> void {
    auto D = dotOp.getD();
    OpBuilder builder(dotOp);
    NvidiaMmaEncodingAttr mmaLayout =
        dyn_cast<NvidiaMmaEncodingAttr>(D.getType().getEncoding());
    if (mmaLayout) {
      int version = mmaLayout.getVersionMajor();
      if (version == 2 || version == 3)
        dotOp.emitError() << "IFWA does not support MMA v" << version;
      else
        dotOp.emitError()
            << "IFWA MMA v" << version
            << " is a declared hardware capability but its compiler lowering "
               "is not connected in this pipeline";
      foundUnsupportedMMA = true;
      return;
    }
    // Residual blocked dots use IFWA FMA lowering. Match operand types to
    // the accumulator exactly as the shared pass did for its FMA case.
    Type AElType = dotOp.getA().getType().getElementType();
    Type promoteType = D.getType().getElementType();
    if (AElType == promoteType)
      return;
    Location loc = dotOp.getLoc();
    Value promotedA = promoteOperand(builder, loc, dotOp.getA(), promoteType);
    Value promotedB = promoteOperand(builder, loc, dotOp.getB(), promoteType);
    dotOp.setOperand(0, promotedA);
    dotOp.setOperand(1, promotedB);
  });
  return failure(foundUnsupportedMMA);
}

// Move a lone RHS scale to the LHS by transposing the matrix dimensions.
static void transposeDotOp(DotScaledOp dotOp) {
  OpBuilder builder(dotOp);
  Value lhs = dotOp.getA();
  int rank = dotOp.getType().getRank();
  SmallVector<int32_t> transOrder = llvm::to_vector(llvm::seq<int32_t>(rank));
  std::swap(transOrder[rank - 2], transOrder[rank - 1]);
  Value lhsTransposed = TransOp::create(builder, lhs.getLoc(), lhs, transOrder);
  Value rhs = dotOp.getB();
  Value rhsTransposed = TransOp::create(builder, rhs.getLoc(), rhs, transOrder);
  Value c = dotOp.getC();
  Value cTransposed = TransOp::create(builder, c.getLoc(), c, transOrder);
  Value result = DotScaledOp::create(
      builder, dotOp.getLoc(), cTransposed.getType(), rhsTransposed,
      lhsTransposed, cTransposed, dotOp.getBScale(), dotOp.getAScale(),
      dotOp.getBElemType(), dotOp.getAElemType(), dotOp.getFastMath(),
      dotOp.getRhsKPack(), dotOp.getLhsKPack());
  Operation *transposedResult =
      TransOp::create(builder, result.getLoc(), result, transOrder);
  dotOp.replaceAllUsesWith(transposedResult);
  dotOp.erase();
}

static void transposeDots(ModuleOp m) {
  // TODO: extend to regular dot when it is profitable.
  SmallVector<DotScaledOp> toTranspose;
  m.walk([&](DotScaledOp dotOp) -> void {
    if (dotOp.getAScale() == nullptr && dotOp.getBScale() != nullptr)
      toTranspose.push_back(dotOp);
  });
  for (DotScaledOp dotOp : toTranspose) {
    transposeDotOp(dotOp);
  }
}

class GPNPUAccelerateMatmulPass
    : public PassWrapper<GPNPUAccelerateMatmulPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPNPUAccelerateMatmulPass)

  StringRef getArgument() const final { return "gpnpu-accelerate-matmul"; }
  StringRef getDescription() const final {
    return "accelerate IFWA matmul with MMA version 5 and blocked FMA";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, triton::TritonDialect,
                gpu::TritonGPUDialect, nvidia_gpu::TritonNvidiaGPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    // We could do this generically if we manage to improve the heuristics
    // reverted in these two PRs https://github.com/triton-lang/triton/pull/5834
    // https://github.com/triton-lang/triton/pull/5837
    transposeDots(m);

    mlir::RewritePatternSet patterns(context);
    constexpr int benefitDefault = 1;
    constexpr int benefitMMAv5 = 10;
    constexpr int benefitSplitBatched = 20;

    populateDecomposeScaledBlockedPatterns(patterns, benefitDefault);
    patterns.add<SplitBatchedScaledDot>(context, benefitSplitBatched);
    patterns.add<BlockedToMMAv5>(context, benefitMMAv5);
    patterns.add<ScaledBlockedToMMAv5>(context, benefitMMAv5);

    if (applyPatternsGreedily(m, std::move(patterns)).failed())
      return signalPassFailure();
    // Now that we have picked the mma type, decompose dot that are not natively
    // supported.
    if (failed(decomposeMixedModeDotOp(m)))
      signalPassFailure();
  }
};

std::unique_ptr<Pass> createGPNPUAccelerateMatmulPass() {
  return std::make_unique<GPNPUAccelerateMatmulPass>();
}

} // namespace Intellifusion
} // namespace triton
} // namespace mlir
