/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "TritonHCUGPUTransforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <limits>
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUGPUOPTIMIZEEPILOGUE
#include "TritonHCUGPUTransforms/Passes.h.inc"

namespace {

bool isOneOperandElementwiseOp(Operation *op) {
  if (llvm::isa<arith::ExtFOp, arith::ExtSIOp, arith::ExtUIOp, arith::FPToSIOp,
                arith::FPToUIOp, arith::NegFOp, arith::SIToFPOp,
                arith::TruncFOp, arith::TruncIOp, arith::UIToFPOp>(op))
    return true;
  if (llvm::isa<math::AbsFOp, math::AbsIOp, math::AtanOp, math::Atan2Op,
                math::CeilOp, math::CosOp, math::SinOp,
                math::CountLeadingZerosOp, math::CountTrailingZerosOp,
                math::CtPopOp, math::ErfOp, math::ExpOp, math::Exp2Op,
                math::ExpM1Op, math::FloorOp, math::LogOp, math::Log10Op,
                math::Log1pOp, math::Log2Op, math::SqrtOp, math::RsqrtOp,
                math::TanhOp>(op))
    return true;
  if (llvm::isa<triton::IntToPtrOp, triton::PtrToIntOp, triton::BitcastOp,
                triton::FpToFpOp>(op))
    return true;
  if (auto externElementwiseOp = dyn_cast<triton::ExternElementwiseOp>(op))
    return op->getNumOperands() == 1 && op->getNumResults() == 1 &&
           externElementwiseOp.getPure();
  return false;
}

// A batch reduction can retain its accumulator slice for the final store.
// Prove that lane/register owners cover the complete result exactly once after
// suppressing zero-basis duplicates, and that every warp holds that same result.
// The normal store lowering already emits the corresponding owner predicate.
static bool isBatchMmacReductionOwnerSlice(Value value,
                                          triton::StoreOp store) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != 2 ||
      (!type.getElementType().isF16() && !type.getElementType().isBF16()))
    return false;
  auto slice = dyn_cast<triton::gpu::SliceEncodingAttr>(type.getEncoding());
  if (!slice || slice.getDim() != 0)
    return false;
  auto mma = dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(slice.getParent());
  if (!mma)
    return false;
  auto module = store->getParentOfType<ModuleOp>();
  if (!module)
    return false;
  auto target = module->getAttrOfType<StringAttr>("ttg.target");
  if (!target || target.getValue() != "hip:gfx936")
    return false;
  auto numWarps = module->getAttrOfType<IntegerAttr>("ttg.num-warps");
  auto isLinearDim = [](int64_t dim) {
    return dim > 0 && dim <= std::numeric_limits<int32_t>::max() &&
           llvm::isPowerOf2_64(static_cast<uint64_t>(dim));
  };
  auto warps = mma.getWarpsPerCTA();
  if (!numWarps || !isLinearDim(numWarps.getInt()) || warps.size() != 3 ||
      warps[0] != numWarps.getInt() || warps[1] != 1 || warps[2] != 1 ||
      triton::gpu::lookupNumWarps(store) != numWarps.getInt() ||
      triton::gpu::lookupNumCTAs(store) != 1 ||
      triton::gpu::TritonGPUDialect::getThreadsPerWarp(module) != 64)
    return false;
  // Keep complete instruction tiles. Instruction legality remains the existing
  // AMD-MFMA verifier/lowering's responsibility, not a model/tile whitelist.
  auto instr = mma.getInstrShape();
  if (instr.size() != 3 || instr[0] == 0 || instr[1] == 0 || instr[2] == 0 ||
      !isLinearDim(type.getShape()[0]) || !isLinearDim(type.getShape()[1]) ||
      type.getShape()[0] >
          std::numeric_limits<int32_t>::max() / type.getShape()[1] ||
      type.getShape()[0] % instr[0] != 0 ||
      type.getShape()[1] % instr[1] != 0)
    return false;

  while (auto *op = value.getDefiningOp()) {
    if (!isOneOperandElementwiseOp(op) || op->getNumOperands() != 1 ||
        op->getNumResults() != 1)
      break;
    auto operandType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (!operandType || operandType.getShape() != type.getShape() ||
        operandType.getEncoding() != slice)
      return false;
    value = op->getOperand(0);
  }
  auto reduce = value.getDefiningOp<triton::ReduceOp>();
  if (!reduce || reduce.getAxis() != 0 || reduce->getNumOperands() != 1 ||
      reduce->getNumResults() != 1)
    return false;
  auto inputType = dyn_cast<RankedTensorType>(reduce->getOperand(0).getType());
  if (!inputType || inputType.getRank() != 3 ||
      inputType.getEncoding() != mma ||
      inputType.getShape().drop_front() != type.getShape() ||
      !isLinearDim(inputType.getShape()[0]) ||
      inputType.getShape()[0] < warps[0] ||
      inputType.getShape()[0] % warps[0] != 0)
    return false;

  auto layout = triton::gpu::toLinearLayout(type);
  auto *ctx = type.getContext();
  auto kRegister = StringAttr::get(ctx, "register");
  auto kLane = StringAttr::get(ctx, "lane");
  auto kWarp = StringAttr::get(ctx, "warp");
  auto kBlock = StringAttr::get(ctx, "block");
  if (!layout.hasInDim(kRegister) || !layout.hasInDim(kLane) ||
      !layout.hasInDim(kWarp) || layout.getNumOutDims() != 2 ||
      layout.getInDimSize(kWarp) != numWarps.getInt() ||
      layout.getInDimSize(kLane) !=
          triton::gpu::TritonGPUDialect::getThreadsPerWarp(module))
    return false;
  for (auto [index, dim] : llvm::enumerate(layout.getOutDimNames())) {
    if (layout.getOutDimSize(dim) != type.getShape()[index])
      return false;
  }
  unsigned ownerBits = 0;
  for (const auto &[dim, bases] : layout.getBases()) {
    if (dim != kRegister && dim != kLane && dim != kWarp && dim != kBlock)
      return false;
    for (const auto &basis : bases) {
      bool nonzero = llvm::any_of(basis, [](int v) { return v != 0; });
      if (dim == kWarp || dim == kBlock) {
        if (nonzero)
          return false;
      } else {
        ownerBits += nonzero;
      }
    }
  }
  auto outDims = llvm::to_vector(layout.getOutDimNames());
  auto owners = layout.sublayout({kRegister, kLane}, outDims);
  // Surjectivity proves coverage. Exactly rank-many nonzero basis vectors
  // proves independence, so only zero bits alias. This is the alias class
  // suppressed by the existing FreeVariableMasks-based store predicate.
  return owners.isSurjective() &&
         ownerBits == owners.getTotalOutDimSizeLog2();
}

// A complete type/layout plan is checked before creating any operation.  The
// intermediate binary axes are all register-only; no Reduce attribute is used
// as a certificate and no general reduction lowering is changed.
struct RegisterBatchOwnerPlan {
  RankedTensorType input, transposed, binary, reversed;
  SmallVector<int32_t> binaryOrder;
  SmallVector<RankedTensorType> splitResults;
  bool sharedVector = false;
};

// Select output register coordinates from the actual accumulator chart first.
// Pure, unique coordinate-unit vectors are independent over GF(2). Completing
// this subset with the unused natural output units proves a full bijection;
// it does not assume a matrix shape, a particular N bit, or native LDS width.
static std::optional<triton::LinearLayout::BasesT>
planSharedVectorOwnerBases(RankedTensorType srcType, int64_t warps) {
  auto shape = srcType.getShape();
  auto *ctx = srcType.getContext();
  auto reg = StringAttr::get(ctx, "register");
  auto lane = StringAttr::get(ctx, "lane");
  auto warp = StringAttr::get(ctx, "warp");
  auto block = StringAttr::get(ctx, "block");
  auto source = triton::gpu::toLinearLayout(srcType);
  if (!source.isInvertible() || source.getNumOutDims() != 3 ||
      !source.hasInDim(reg) || !source.hasInDim(lane) ||
      !source.hasInDim(warp) || !source.hasInDim(block) ||
      source.getInDimSize(lane) != 64 ||
      source.getInDimSize(warp) != warps || source.getInDimSize(block) != 1)
    return std::nullopt;
  for (auto [i, dim] : llvm::enumerate(source.getOutDimNames()))
    if (dim != StringAttr::get(ctx, "dim" + std::to_string(i)) ||
        source.getOutDimSize(dim) != shape[i])
      return std::nullopt;

  using Basis = std::vector<int32_t>;
  SmallVector<Basis> natural, sourceRegisters, selected;
  // Natural output order is N-fast, as in the original owner plan.
  for (unsigned dim : {2u, 1u})
    for (int64_t bit = 1; bit < shape[dim]; bit *= 2) {
      Basis unit(3, 0);
      unit[dim] = static_cast<int32_t>(bit);
      natural.push_back(std::move(unit));
    }
  for (const auto &[dim, vectors] : source.getBases()) {
    if (dim != reg)
      continue;
    for (const auto &vector : vectors) {
      if (vector.size() != 3 || vector[0] != 0)
        return std::nullopt;
      if (llvm::all_of(vector, [](int32_t v) { return v == 0; }))
        continue; // Redundant zero bits cannot supply an output coordinate.
      Basis unit(vector.begin(), vector.end());
      if (!llvm::is_contained(natural, unit) ||
          llvm::is_contained(sourceRegisters, unit))
        return std::nullopt;
      sourceRegisters.push_back(std::move(unit));
    }
  }
  const int64_t perThread = shape[1] * shape[2] / (64 * warps);
  const unsigned registerBits = llvm::Log2_64(perThread);
  for (const auto &unit : sourceRegisters) {
    if (selected.size() == registerBits)
      break;
    selected.push_back(unit);
  }
  for (const auto &unit : natural) {
    if (selected.size() == registerBits)
      break;
    if (!llvm::is_contained(selected, unit))
      selected.push_back(unit);
  }
  if (selected.size() != registerBits)
    return std::nullopt;

  triton::LinearLayout::BasesT bases;
  bases[reg];
  for (int64_t bit = 1; bit < shape[0]; bit *= 2)
    bases[reg].push_back({static_cast<int32_t>(bit), 0, 0});
  for (const auto &unit : selected)
    bases[reg].push_back(unit);
  unsigned laneBits = 0, warpBits = 0;
  for (const auto &unit : natural) {
    if (llvm::is_contained(selected, unit))
      continue;
    if (laneBits < 6) {
      bases[lane].push_back(unit);
      ++laneBits;
    } else {
      bases[warp].push_back(unit);
      ++warpBits;
    }
  }
  bases[warp]; // Preserve the size-one dimension when numWarps is one.
  bases[block];
  if (laneBits != 6 || warpBits != llvm::Log2_64(warps))
    return std::nullopt;
  return bases;
}

static std::optional<RegisterBatchOwnerPlan>
planRegisterBatchOwner(RankedTensorType srcType, int64_t warps, Location loc,
                       bool sharedVector = false) {
  auto shape = srcType.getShape();
  const int64_t batch = shape[0], rows = shape[1], cols = shape[2];
  const int64_t threads = 64 * warps;
  const int64_t outputs = rows * cols;
  // The existing full-wave backend reduction has a different tree.  This
  // subset corresponds exactly to its ordinary descending-XOR fallback.
  if (batch >= 64 || outputs < threads || outputs % threads)
    return std::nullopt;

  auto *ctx = srcType.getContext();
  auto reg = StringAttr::get(ctx, "register");
  auto lane = StringAttr::get(ctx, "lane");
  auto warp = StringAttr::get(ctx, "warp");
  auto block = StringAttr::get(ctx, "block");
  triton::LinearLayout::BasesT bases;
  for (int64_t bit = 1; bit < batch; bit *= 2)
    bases[reg].push_back({static_cast<int32_t>(bit), 0, 0});
  auto outputBits = [&](StringAttr dim, int64_t size, int64_t stride) {
    auto &vectors = bases[dim];
    for (int64_t bit = 1; bit < size; bit *= 2) {
      int64_t flat = bit * stride;
      vectors.push_back({0, static_cast<int32_t>(flat / cols),
                         static_cast<int32_t>(flat % cols)});
    }
  };
  if (sharedVector) {
    auto sharedBases = planSharedVectorOwnerBases(srcType, warps);
    if (!sharedBases)
      return std::nullopt;
    bases = std::move(*sharedBases);
  } else {
    outputBits(reg, outputs / threads, threads);
    outputBits(lane, 64, 1);
    outputBits(warp, warps, 64);
    outputBits(block, 1, 1);
  }
  SmallVector<std::pair<StringAttr, int32_t>> outDims{
      {StringAttr::get(ctx, "dim0"), static_cast<int32_t>(batch)},
      {StringAttr::get(ctx, "dim1"), static_cast<int32_t>(rows)},
      {StringAttr::get(ctx, "dim2"), static_cast<int32_t>(cols)}};
  triton::LinearLayout ll(std::move(bases), outDims,
                         /*requireSurjective=*/true);
  if (!ll.isInvertible() || ll.getInDimSize(lane) != 64 ||
      ll.getInDimSize(warp) != warps || ll.getInDimSize(block) != 1)
    return std::nullopt;

  RegisterBatchOwnerPlan plan;
  plan.sharedVector = sharedVector;
  auto encoding = triton::gpu::LinearEncodingAttr::get(ctx, ll);
  plan.input = srcType.cloneWithEncoding(encoding);
  auto *infer =
      cast<triton::DialectInferLayoutInterface>(&encoding.getDialect());
  auto inferTrans = [&](RankedTensorType type, ArrayRef<int32_t> order)
      -> RankedTensorType {
    Attribute resultEncoding;
    if (failed(infer->inferTransOpEncoding(type.getEncoding(), type.getShape(),
                                          order, resultEncoding, loc)))
      return {};
    SmallVector<int64_t> resultShape;
    for (int32_t dim : order)
      resultShape.push_back(type.getShape()[dim]);
    return RankedTensorType::get(resultShape, type.getElementType(),
                                 resultEncoding);
  };
  plan.transposed = inferTrans(plan.input, {1, 2, 0});
  if (!plan.transposed)
    return std::nullopt;
  SmallVector<int64_t> binaryShape{rows, cols};
  for (int64_t n = batch; n > 1; n /= 2)
    binaryShape.push_back(2);
  Attribute binaryEncoding;
  if (failed(infer->inferReshapeOpEncoding(
          plan.transposed.getShape(), plan.transposed.getEncoding(),
          binaryShape, binaryEncoding, loc)))
    return std::nullopt;
  plan.binary = RankedTensorType::get(binaryShape, srcType.getElementType(),
                                     binaryEncoding);
  if (triton::gpu::isExpensiveView(plan.transposed, plan.binary))
    return std::nullopt;
  plan.binaryOrder = {0, 1};
  for (int32_t dim = static_cast<int32_t>(binaryShape.size()) - 1; dim >= 2;
       --dim)
    plan.binaryOrder.push_back(dim);
  plan.reversed = inferTrans(plan.binary, plan.binaryOrder);
  if (!plan.reversed)
    return std::nullopt;

  auto type = plan.reversed;
  while (type.getRank() > 2) {
    if (type.getShape().back() != 2)
      return std::nullopt;
    auto splitLL = triton::gpu::toLinearLayout(type);
    if (!splitLL.isInvertible())
      return std::nullopt;
    unsigned unitRegisterBits = 0;
    for (const auto &[dim, vectors] : splitLL.getBases()) {
      for (const auto &basis : vectors) {
        if (basis.back() == 0)
          continue;
        // Exactly one pure register bit owns the split axis.  Neither lane,
        // warp nor another output coordinate may be mixed into that bit.
        if (dim != reg || basis.back() != 1 ||
            llvm::any_of(ArrayRef<int32_t>(basis).drop_back(),
                         [](int32_t v) { return v != 0; }))
          return std::nullopt;
        ++unitRegisterBits;
      }
    }
    if (unitRegisterBits != 1)
      return std::nullopt;
    Attribute nextEncoding;
    if (failed(infer->inferSplitOpEncoding(type.getEncoding(), nextEncoding,
                                          type.getShape(), loc)))
      return std::nullopt;
    auto next = RankedTensorType::get(type.getShape().drop_back(),
                                     type.getElementType(), nextEncoding);
    auto nextLL = triton::gpu::toLinearLayout(next);
    if (!nextLL.isInvertible() ||
        nextLL.getInDimSize(reg) * 2 != splitLL.getInDimSize(reg) ||
        nextLL.getInDimSize(lane) != 64 ||
        nextLL.getInDimSize(warp) != warps ||
        nextLL.getInDimSize(block) != 1)
      return std::nullopt;
    plan.splitResults.push_back(next);
    type = next;
  }
  if (type.getShape() != shape.drop_front() ||
      triton::gpu::getTotalElemsPerThread(type) != outputs / threads)
    return std::nullopt;
  return plan;
}

// Move a single all-warp batch reduction to a proved owner chart.  The opt-in
// branch reifies the exact tree in registers; the original lane-subgroup
// ConvertLayout/Reduce path remains the fallback. No wait/barrier is erased.
class WarpSynchronousOwnerReduction
    : public OpRewritePattern<triton::StoreOp> {
public:
  explicit WarpSynchronousOwnerReduction(MLIRContext *ctx)
      : OpRewritePattern(ctx, PatternBenefit(2)) {}

  LogicalResult matchAndRewrite(triton::StoreOp store,
                                PatternRewriter &rewriter) const override {
    auto resultType = dyn_cast<RankedTensorType>(store.getValue().getType());
    auto ptrType = dyn_cast<RankedTensorType>(store.getPtr().getType());
    if (!resultType || !ptrType || resultType.getRank() != 2 ||
        ptrType.getShape() != resultType.getShape() ||
        (!resultType.getElementType().isBF16() &&
         !resultType.getElementType().isF16()))
      return failure();
    if (store.getMask()) {
      auto maskType = dyn_cast<RankedTensorType>(store.getMask().getType());
      if (!maskType || maskType.getShape() != resultType.getShape())
        return failure();
    }

    // Only this one store consumes the reduced tensor, through optional data
    // layout conversions and a strict unary whitelist. No load, binary op,
    // shared value or side-effecting consumer may enter the trace.
    SmallVector<Operation *> chain;
    Value value = store.getValue();
    while (auto *op = value.getDefiningOp()) {
      if (isa<triton::ReduceOp>(op))
        break;
      if (op->getBlock() != store->getBlock() || !op->hasOneUse() ||
          op->getNumOperands() != 1 || op->getNumResults() != 1 ||
          (!isa<triton::gpu::ConvertLayoutOp>(op) &&
           !isOneOperandElementwiseOp(op)))
        return failure();
      auto inType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto outType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
      if (!inType || !outType ||
          inType.getShape() != resultType.getShape() ||
          outType.getShape() != resultType.getShape())
        return failure();
      chain.push_back(op);
      value = op->getOperand(0);
    }
    auto reduce = value.getDefiningOp<triton::ReduceOp>();
    if (!reduce || !reduce->hasOneUse() || reduce.getAxis() != 0 ||
        reduce->getNumOperands() != 1 || reduce->getNumResults() != 1 ||
        !reduce.getCombineOp().hasOneBlock())
      return failure();
    // Clone this exact f32 add region, including attributes/fastmath flags.
    // Restrict to two block args, one ordered add and its return: no additional
    // arithmetic is commuted, reassociated, or duplicated by this rewrite.
    Block &combine = reduce.getCombineOp().front();
    if (combine.getNumArguments() != 2 ||
        std::distance(combine.begin(), combine.end()) != 2)
      return failure();
    auto add = dyn_cast_or_null<arith::AddFOp>(reduce.getSingleCombiner());
    auto ret = dyn_cast<triton::ReduceReturnOp>(combine.getTerminator());
    if (!add || !ret || ret->getNumOperands() != 1 ||
        ret->getOperand(0) != add.getResult() ||
        !combine.getArgument(0).getType().isF32() ||
        !combine.getArgument(1).getType().isF32())
      return failure();

    auto srcType = dyn_cast<RankedTensorType>(reduce->getOperand(0).getType());
    if (!srcType || srcType.getRank() != 3 ||
        !srcType.getElementType().isF32() ||
        srcType.getShape().drop_front() != resultType.getShape())
      return failure();
    auto mma = dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(srcType.getEncoding());
    auto module = store->getParentOfType<ModuleOp>();
    if (!mma || !module || reduce->getBlock() != store->getBlock())
      return failure();
    auto target = module->getAttrOfType<StringAttr>("ttg.target");
    if (!target || target.getValue() != "hip:gfx936" ||
        triton::gpu::lookupNumCTAs(store) != 1 ||
        triton::gpu::TritonGPUDialect::getThreadsPerWarp(module) != 64)
      return failure();
    auto isRepresentable = [](int64_t v) {
      return v > 0 && v <= std::numeric_limits<int32_t>::max() &&
             llvm::isPowerOf2_64(static_cast<uint64_t>(v));
    };
    const int64_t batch = srcType.getShape()[0];
    const int64_t rows = srcType.getShape()[1];
    const int64_t cols = srcType.getShape()[2];
    auto moduleWarps = module->getAttrOfType<IntegerAttr>("ttg.num-warps");
    if (!moduleWarps || !isRepresentable(moduleWarps.getInt()))
      return failure();
    const int64_t warps = triton::gpu::lookupNumWarps(store);
    if (!isRepresentable(batch) || !isRepresentable(rows) ||
        !isRepresentable(cols) || !isRepresentable(warps) ||
        warps != moduleWarps.getInt() || warps <= 1 || warps > 64 ||
        batch != warps ||
        rows > std::numeric_limits<int32_t>::max() / cols ||
        rows * cols > std::numeric_limits<int32_t>::max() / batch)
      return failure();
    auto wpc = mma.getWarpsPerCTA();
    auto cta = mma.getCTALayout();
    auto instr = mma.getInstrShape();
    if (wpc.size() != 3 || wpc[0] != warps || wpc[1] != 1 || wpc[2] != 1 ||
        !llvm::all_of(cta.getCTAsPerCGA(), [](unsigned v) { return v == 1; }) ||
        !llvm::all_of(cta.getCTASplitNum(), [](unsigned v) { return v == 1; }) ||
        instr.size() != 3 || !instr[0] || !instr[1] || !instr[2] ||
        rows % instr[0] || cols % instr[1])
      return failure();

    auto *ctx = store.getContext();
    const auto kReg = StringAttr::get(ctx, "register");
    const auto kLane = StringAttr::get(ctx, "lane");
    const auto kWarp = StringAttr::get(ctx, "warp");
    const auto kBlock = StringAttr::get(ctx, "block");
    auto oldLL = triton::gpu::toLinearLayout(srcType);
    // Check the actual layout, not just its attribute name: one batch value
    // per physical warp, natural batch bits on warp, no lane/register batch
    // fold. Hence the old path has no intra-thread or intra-warp FP32 add;
    // its final warpReduce(batch, interleave=1) has ordered batch IDs. The
    // new intra-warp call uses the same backend tree (including full-wave DPP
    // if applicable), not an independently reassociated sum.
    if (!oldLL.hasInDim(kReg) || !oldLL.hasInDim(kLane) ||
        !oldLL.hasInDim(kWarp) || !oldLL.hasInDim(kBlock) ||
        !oldLL.isInvertible() || oldLL.getInDimSize(kLane) != 64 ||
        oldLL.getInDimSize(kWarp) != warps ||
        oldLL.getInDimSize(kBlock) != 1)
      return failure();
    for (const auto &[dim, bases] : oldLL.getBases()) {
      if (dim != kReg && dim != kLane && dim != kWarp && dim != kBlock)
        return failure();
      for (auto [i, basis] : llvm::enumerate(bases)) {
        if (basis.size() != 3)
          return failure();
        if (dim == kWarp) {
          if (basis[0] != (int64_t{1} << i) || basis[1] || basis[2])
            return failure();
        } else if (basis[0]) {
          return failure();
        }
      }
    }

    // The option only requests the alternative; all source/tree/layout gates
    // above and the complete register-only type plan below are independent.
    std::optional<RegisterBatchOwnerPlan> registerPlan;
    auto requested = module->getAttrOfType<BoolAttr>(
        "ttg.hcu.register_batch_owner");
    if (requested && requested.getValue() &&
        add.getLhs() == combine.getArgument(0) &&
        add.getRhs() == combine.getArgument(1)) {
      auto sharedRequested = module->getAttrOfType<BoolAttr>(
          "ttg.hcu.register_batch_owner_shared_vector");
      if (sharedRequested && sharedRequested.getValue())
        registerPlan = planRegisterBatchOwner(srcType, warps, reduce.getLoc(),
                                               /*sharedVector=*/true);
      // Failed source-basis/type/split proof falls back to the unchanged v1
      // register-owner plan before any operation is created.
      if (!registerPlan)
        registerPlan = planRegisterBatchOwner(srcType, warps, reduce.getLoc());
    }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(reduce);
    Value newValue;
    Attribute newEncoding;
    if (registerPlan) {
      const auto &plan = *registerPlan;
      Value tree = triton::gpu::ConvertLayoutOp::create(
          rewriter, reduce.getLoc(), plan.input, reduce->getOperand(0));
      tree = triton::TransOp::create(
          rewriter, reduce.getLoc(), tree, SmallVector<int32_t>{1, 2, 0});
      tree = triton::ReshapeOp::create(
          rewriter, reduce.getLoc(), plan.binary.getShape(), tree,
          /*allowReorder=*/false);
      tree = triton::TransOp::create(rewriter, reduce.getLoc(), tree,
                                    plan.binaryOrder);
      // Original batch bits are split from high to low. For B=4 the first
      // tensor add is [x0+x2, x1+x3], and the second is their ordered sum.
      // Existing Trans/Reshape/Split lowerings just repack this proved
      // thread-local register set. Clone AddFOp, including all properties and
      // attributes; do not turn an arbitrary Reduce into a balanced reduction.
      for (auto type : plan.splitResults) {
        auto split = triton::SplitOp::create(rewriter, reduce.getLoc(), tree);
        IRMapping mapping;
        mapping.map(combine.getArgument(0), split.getResult(0));
        mapping.map(combine.getArgument(1), split.getResult(1));
        auto *orderedAdd = rewriter.clone(*add.getOperation(), mapping);
        orderedAdd->getResult(0).setType(type);
        tree = orderedAdd->getResult(0);
      }
      newValue = tree;
      newEncoding = cast<RankedTensorType>(tree.getType()).getEncoding();
      auto count = module->getAttrOfType<IntegerAttr>(
          "ttg.hcu.register_batch_owner_count");
      int64_t countValue =
          count && count.getType().isSignlessInteger(32) ? count.getInt() : 0;
      int64_t nextCount = countValue >= 0 &&
                                  countValue < std::numeric_limits<int32_t>::max()
                              ? countValue + 1
                              : 1;
      rewriter.modifyOpInPlace(module, [&] {
        module->setAttr("ttg.hcu.register_batch_owner_count",
                        rewriter.getI32IntegerAttr(nextCount));
        if (plan.sharedVector) {
          auto sharedCount = module->getAttrOfType<IntegerAttr>(
              "ttg.hcu.register_batch_owner_shared_vector_count");
          int64_t value = sharedCount && sharedCount.getType().isSignlessInteger(32)
                              ? sharedCount.getInt() : 0;
          int64_t next = value >= 0 && value < std::numeric_limits<int32_t>::max()
                             ? value + 1 : 1;
          module->setAttr("ttg.hcu.register_batch_owner_shared_vector_count",
                          rewriter.getI32IntegerAttr(next));
        }
      });
    } else {
      // Binary flattening: batch is fastest, then N, then M. Lane low bits
      // enumerate the entire reduction; warp/register bits enumerate outputs.
      // This is bijective before reduction, and Slice(0) has exactly the batch
      // lane bits as zero bases. Existing store owner predicates suppress only
      // those duplicates; output ownership remains distributed across warps.
      const int64_t threads = 64 * warps;
      const int64_t elements = batch * rows * cols;
      if (elements < threads || elements % threads)
        return failure();
      triton::LinearLayout::BasesT bases;
      auto appendBits = [&](StringAttr dim, int64_t size, int64_t stride) {
        auto &vectors = bases[dim]; // Include even size-one input dimensions.
        for (int64_t bit = 1; bit < size; bit *= 2) {
          int64_t flat = bit * stride;
          vectors.push_back({static_cast<int32_t>(flat % batch),
                             static_cast<int32_t>(flat / (batch * cols)),
                             static_cast<int32_t>((flat / batch) % cols)});
        }
      };
      appendBits(kReg, elements / threads, threads);
      appendBits(kLane, 64, 1);
      appendBits(kWarp, warps, 64);
      appendBits(kBlock, 1, 1);
      SmallVector<std::pair<StringAttr, int32_t>> outDims{
          {StringAttr::get(ctx, "dim0"), static_cast<int32_t>(batch)},
          {StringAttr::get(ctx, "dim1"), static_cast<int32_t>(rows)},
          {StringAttr::get(ctx, "dim2"), static_cast<int32_t>(cols)}};
      triton::LinearLayout newLL(std::move(bases), outDims,
                                 /*requireSurjective=*/true);
      auto linear = triton::gpu::LinearEncodingAttr::get(ctx, newLL);
      // A second explicit check keeps the lowering contract local to the pattern.
      if (triton::gpu::getWarpsPerCTA(linear, srcType.getShape())[0] != 1 ||
          triton::gpu::getThreadsPerWarp(linear, srcType.getShape())[0] != batch)
        return failure();

      auto converted = triton::gpu::ConvertLayoutOp::create(
          rewriter, reduce.getLoc(), srcType.cloneWithEncoding(linear),
          reduce->getOperand(0));
      auto newReduce = triton::ReduceOp::create(
          rewriter, reduce.getLoc(), ValueRange{converted.getResult()}, 0);
      newReduce->setAttrs(reduce->getAttrs());
      rewriter.cloneRegionBefore(reduce.getCombineOp(), newReduce.getCombineOp(),
                                 newReduce.getCombineOp().end());
      newValue = newReduce->getResult(0);
      newEncoding = cast<RankedTensorType>(newValue.getType()).getEncoding();
    }
    // Clone the pure suffix at the store. Skip data layout conversions: the
    // reduced value already has its final owner encoding. Pointer/mask use
    // that encoding, preserving every logical pointer and guard value.
    rewriter.setInsertionPoint(store);
    for (auto *op : llvm::reverse(chain)) {
      if (isa<triton::gpu::ConvertLayoutOp>(op))
        continue;
      IRMapping mapping;
      mapping.map(op->getOperand(0), newValue);
      auto *copy = rewriter.clone(*op, mapping);
      copy->getResult(0).setType(
          cast<RankedTensorType>(op->getResult(0).getType())
              .cloneWithEncoding(newEncoding));
      newValue = copy->getResult(0);
    }
    auto newPtr = triton::gpu::ConvertLayoutOp::create(
        rewriter, store.getLoc(), ptrType.cloneWithEncoding(newEncoding),
        store.getPtr());
    IRMapping storeMapping;
    storeMapping.map(store.getPtr(), newPtr.getResult());
    storeMapping.map(store.getValue(), newValue);
    if (store.getMask()) {
      auto maskType = cast<RankedTensorType>(store.getMask().getType());
      auto newMask = triton::gpu::ConvertLayoutOp::create(
          rewriter, store.getLoc(), maskType.cloneWithEncoding(newEncoding),
          store.getMask());
      storeMapping.map(store.getMask(), newMask.getResult());
    }
    rewriter.clone(*store.getOperation(), storeMapping); // Keep all store attrs.
    rewriter.eraseOp(store);
    for (auto *op : chain)
      rewriter.eraseOp(op); // All trace nodes were single-use; now dead.
    rewriter.eraseOp(reduce);
    return success();
  }
};

// Tries to optimize oldStoreOp with v_permlane*_swap instruction when possible.
// Returns null store op if not suitable.
static triton::StoreOp
usePermlaneSwapToOptimizeStore(PatternRewriter &rewriter, Value ptr, Value val,
                               Value mask, triton::StoreOp oldStoreOp) {
  auto ptrType = cast<RankedTensorType>(ptr.getType());
  auto valType = cast<RankedTensorType>(val.getType());

  // Create a new layout where each thread holds 8 consecutive elements, in
  // order to enable wide 128-bit global stores.
  std::optional<triton::LinearLayout> storeLL =
      triton::gpu::chooseMfmaLikeStoreLayout(valType);
  if (!storeLL)
    return nullptr;

  Attribute newEncoding = triton::gpu::LinearEncodingAttr::get(
      oldStoreOp.getContext(), storeLL.value());
  auto newPtrType = ptrType.cloneWithEncoding(newEncoding);
  Value newPtr = triton::gpu::ConvertLayoutOp::create(rewriter, ptr.getLoc(),
                                                      newPtrType, ptr);

  auto newValType = valType.cloneWithEncoding(newEncoding);
  Value newVal = triton::gpu::ConvertLayoutOp::create(rewriter, val.getLoc(),
                                                      newValType, val);

  Value newMask = mask;
  if (mask) {
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    auto newMaskType = maskType.cloneWithEncoding(newEncoding);
    newMask = triton::gpu::ConvertLayoutOp::create(rewriter, mask.getLoc(),
                                                   newMaskType, mask);
  }

  return triton::StoreOp::create(rewriter, oldStoreOp.getLoc(), newPtr, newVal,
                                 newMask, oldStoreOp.getCache(),
                                 oldStoreOp.getEvict());
}

// convert(val) : xmma -> blocked
// elementWiseOp(val) : blocked
// ...
// elementWiseOp(val) : blocked
// tt.store(ptr, val, mask, ...) : blocked
// ==>
// convert(ptr) : blocked -> xmma
// convert(mask) : blocked -> xmma
// elementWiseOp(val) : xmma
// ...
// elementWiseOp(val) : xmma
// tt.store(ptr, val, mask, ...) : xmma
//
// Store with xmma layout directly
//
// xmma layout is either MFMA or WMMA
class BypassEpilogueSMEM : public mlir::OpRewritePattern<triton::StoreOp> {

public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp stOp,
                  mlir::PatternRewriter &rewriter) const override {

    Value ptr = stOp.getPtr();
    Value val = stOp.getValue();
    Value mask = stOp.getMask();
    auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
    auto valType = dyn_cast<RankedTensorType>(val.getType());
    if (!ptrType || !valType ||
        !isa<triton::gpu::BlockedEncodingAttr>(ptrType.getEncoding()) ||
        !isa<triton::gpu::BlockedEncodingAttr>(valType.getEncoding()))
      return mlir::failure();

    llvm::SmallVector<mlir::Operation *> chainedOps;
    while (true) {
      auto chainedOp = val.getDefiningOp();
      if (!chainedOp)
        return mlir::failure();
      if (llvm::isa<triton::gpu::ConvertLayoutOp>(chainedOp))
        break;
      if (!chainedOp->hasOneUse() || chainedOp->getNumOperands() != 1 ||
          chainedOp->getNumResults() != 1)
        return mlir::failure();
      if (!isOneOperandElementwiseOp(chainedOp))
        return mlir::failure();
      val = chainedOp->getOperand(0);
      chainedOps.push_back(chainedOp);
    }

    auto cvtOp = val.getDefiningOp<triton::gpu::ConvertLayoutOp>();
    if (!cvtOp)
      return mlir::failure();

    auto encoding = cvtOp.getSrc().getType().getEncoding();
    if (!isa<triton::gpu::MmaEncodingTrait>(encoding) &&
        !isBatchMmacReductionOwnerSlice(cvtOp.getSrc(), stOp))
      return mlir::failure();

    if (!cvtOp.getResult().hasOneUse())
      return mlir::failure();

    auto newEncoding =
        cast<RankedTensorType>(cvtOp.getSrc().getType()).getEncoding();

    auto newPtrType = ptrType.cloneWithEncoding(newEncoding);
    Value newPtr = triton::gpu::ConvertLayoutOp::create(rewriter, ptr.getLoc(),
                                                        newPtrType, ptr);

    auto newVal = cvtOp.getSrc();

    for (auto chainedOp : llvm::reverse(chainedOps)) {
      auto oldType =
          cast<mlir::RankedTensorType>(chainedOp->getResult(0).getType());
      chainedOp->setOperand(0, newVal);
      newVal = llvm::cast<mlir::TypedValue<RankedTensorType>>(
          chainedOp->getResult(0));

      auto newType = oldType.cloneWithEncoding(newEncoding);
      newVal.setType(newType);
    }

    Value newMask = mask;
    if (mask) {
      auto maskType = dyn_cast<RankedTensorType>(mask.getType());
      auto newMaskType = maskType.cloneWithEncoding(newEncoding);
      newMask = triton::gpu::ConvertLayoutOp::create(rewriter, mask.getLoc(),
                                                     newMaskType, mask);
    }
    triton::StoreOp newStoreOp =
        usePermlaneSwapToOptimizeStore(rewriter, newPtr, newVal, newMask, stOp);
    if (!newStoreOp) {
      newStoreOp =
          triton::StoreOp::create(rewriter, stOp.getLoc(), newPtr, newVal,
                                  newMask, stOp.getCache(), stOp.getEvict());
    }

    rewriter.replaceOp(stOp, newStoreOp);
    return mlir::success();
  }
};

} // anonymous namespace

class TritonHCUGPUOptimizeEpiloguePass
    : public impl::TritonHCUGPUOptimizeEpilogueBase<
          TritonHCUGPUOptimizeEpiloguePass> {

public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<WarpSynchronousOwnerReduction>(context);
    patterns.add<BypassEpilogueSMEM>(context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }
  }
};

} // namespace mlir
