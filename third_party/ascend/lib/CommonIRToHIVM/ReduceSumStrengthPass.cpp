/*
 * Copyright 2026- Xcoresigma Technology Co., Ltd
 */

#include "ascend/include/CommonIRToHIVM/ReduceSumStrengthPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DECL_REDUCESUMSTRENGTH
#define GEN_PASS_DEF_REDUCESUMSTRENGTH
#include "ascend/include/CommonIRToHIVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

enum class CostModelDType { F32, F16, BF16, I32, Unsupported };

CostModelDType getCostModelDType(Type elementType) {
  if (isa<Float32Type>(elementType))
    return CostModelDType::F32;
  if (isa<Float16Type>(elementType))
    return CostModelDType::F16;
  if (isa<BFloat16Type>(elementType))
    return CostModelDType::BF16;
  if (auto intType = dyn_cast<IntegerType>(elementType)) {
    if (intType.getWidth() == 32)
      return CostModelDType::I32;
  }
  return CostModelDType::Unsupported;
}

bool isAddCombiner(ReduceOp reduceOp, Type elementType) {
  Region &combineOp = reduceOp.getCombineOp();
  if (!combineOp.hasOneBlock() || combineOp.front().getOperations().size() != 2)
    return false;

  Operation *combiner = reduceOp.getSingleCombiner();
  if (!combiner)
    return false;

  if (isa<FloatType>(elementType))
    return isa<arith::AddFOp>(combiner);
  if (isa<IntegerType>(elementType))
    return isa<arith::AddIOp>(combiner);
  return false;
}

struct CostModelEntry {
  CostModelDType dtype;
  int64_t minProfitableReduceDim;
  int64_t maxProfitableReduceDim;
};

const CostModelEntry costModel[] = {
    {CostModelDType::F32, 256, 16384},
    {CostModelDType::F16, 512, 8192},
    {CostModelDType::BF16, 512, 512},
    {CostModelDType::I32, 0, -1},
};

bool isProfitable(Type elementType, int64_t reduceDim) {
  CostModelDType dtype = getCostModelDType(elementType);
  for (const CostModelEntry &entry : costModel) {
    if (entry.dtype != dtype)
      continue;
    return reduceDim >= entry.minProfitableReduceDim &&
           reduceDim <= entry.maxProfitableReduceDim &&
           llvm::isPowerOf2_64(reduceDim);
  }
  return false;
}

class ReduceSumStrengthPattern : public OpRewritePattern<ReduceOp> {
public:
  explicit ReduceSumStrengthPattern(MLIRContext *context)
      : OpRewritePattern(context) {}

  LogicalResult matchAndRewrite(ReduceOp reduceOp,
                                PatternRewriter &rewriter) const override {
    if (reduceOp.getNumOperands() != 1 || reduceOp->getNumResults() != 1)
      return failure();

    auto inputType =
        dyn_cast<RankedTensorType>(reduceOp.getOperand(0).getType());
    if (!inputType || !inputType.hasStaticShape())
      return failure();

    Type elementType = inputType.getElementType();
    if (!isAddCombiner(reduceOp, elementType))
      return failure();

    int64_t axis = reduceOp.getAxis();
    if (axis < 0 || axis >= inputType.getRank())
      return failure();

    ArrayRef<int64_t> shape = inputType.getShape();
    int64_t reduceDim = shape[axis];
    if (reduceDim <= 1 || reduceDim % 2 != 0)
      return failure();

    if (!isProfitable(elementType, reduceDim))
      return failure();

    SmallVector<int64_t> halfShape(shape.begin(), shape.end());
    halfShape[axis] = reduceDim / 2;
    auto halfType =
        RankedTensorType::get(halfShape, elementType, inputType.getEncoding());

    SmallVector<OpFoldResult> loOffsets(inputType.getRank(),
                                        rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> hiOffsets(inputType.getRank(),
                                        rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    SmallVector<OpFoldResult> strides(inputType.getRank(),
                                      rewriter.getIndexAttr(1));
    sizes.reserve(inputType.getRank());
    for (int64_t dim = 0; dim < inputType.getRank(); ++dim)
      sizes.push_back(rewriter.getIndexAttr(halfShape[dim]));
    hiOffsets[axis] = rewriter.getIndexAttr(reduceDim / 2);

    Location loc = reduceOp.getLoc();
    Value input = reduceOp.getOperand(0);
    auto lo = rewriter.create<tensor::ExtractSliceOp>(
        loc, halfType, input, loOffsets, sizes, strides);
    auto hi = rewriter.create<tensor::ExtractSliceOp>(
        loc, halfType, input, hiOffsets, sizes, strides);

    Value preAdded;
    if (isa<FloatType>(elementType))
      preAdded = rewriter.create<arith::AddFOp>(loc, lo, hi);
    else if (isa<IntegerType>(elementType))
      preAdded = rewriter.create<arith::AddIOp>(loc, lo, hi);
    else
      return failure();

    auto newReduce = rewriter.create<ReduceOp>(loc, ValueRange{preAdded}, axis);
    for (auto attr : reduceOp->getAttrs()) {
      if (!newReduce->hasAttr(attr.getName()))
        newReduce->setAttr(attr.getName(), attr.getValue());
    }

    Region &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(reduceOp.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(reduceOp, newReduce.getResults());
    return success();
  }
};

class ReduceSumStrengthPass
    : public mlir::triton::impl::ReduceSumStrengthBase<ReduceSumStrengthPass> {
public:
  using mlir::triton::impl::ReduceSumStrengthBase<
      ReduceSumStrengthPass>::ReduceSumStrengthBase;

  void runOnOperation() override {
    if (!enable)
      return;
    if (splitFactor != 2) {
      getOperation().emitError(
          "reduce-sum-strength only supports split-factor=2");
      signalPassFailure();
      return;
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ReduceSumStrengthPattern>(context);

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createReduceSumStrengthPass() {
  return std::make_unique<ReduceSumStrengthPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createReduceSumStrengthPass(bool enable, int32_t splitFactor) {
  ReduceSumStrengthOptions opts;
  opts.enable = enable;
  opts.splitFactor = splitFactor;
  return std::make_unique<ReduceSumStrengthPass>(opts);
}

} // namespace triton
} // namespace mlir
