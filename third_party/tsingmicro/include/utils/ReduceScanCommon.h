#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <numeric>

namespace mlir {
namespace triton {

// Base class for converting scans and reductions.
//
// It provides accumulation function that clones operations from the
// original combine region and applies them on provided tensor.
// Also, it handles multi-dimensional cases reducing them to two
// possible options: lowering for a 1-D tensor inputs and lowering
// the operation over the leading dimension.
//
// Specialized pattern should implement lower1DInput to handle
// trailing dimension case and lowerLeadingDimension to handle the leading
// dimension case through accumulation of sub-tensors.
template <typename OpT, typename ReturnOpT>
struct ReduceScanOpConversionBase : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpConversionPattern<OpT>::getTypeConverter;
  using typename OpConversionPattern<OpT>::OpAdaptor;

  virtual SmallVector<Value>
  lower1DInput(ValueRange inputs, OpT op,
               ConversionPatternRewriter &rewriter) const = 0;
  virtual SmallVector<Value>
  lowerLeadingDimension(ValueRange inputs, OpT op,
                        ConversionPatternRewriter &rewriter) const = 0;

  virtual uint32_t getAxis(OpT op) const = 0;

  virtual SmallVector<Value> getInputs(OpT op) const = 0;

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rank = cast<RankedTensorType>(op.getOperand(0).getType()).getRank();
    auto axis = getAxis(op);
    assert(axis < rank && "Expected axis is within the input rank");
    if (axis == (rank - 1))
      return lowerTrailingDimension(op, rewriter);

    return lowerNonTrailingDimension(op, rewriter);
  }

  // To handle the trailing dimension case, we extract all input vectors
  // and process them through lower1DInput, then build the resulting
  // vector using inserts.
  LogicalResult
  lowerTrailingDimension(OpT op, ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    SmallVector<Value> inputs;
    if (failed(rewriter.getRemappedValues(getInputs(op), inputs)))
      return failure();

    auto inputType = cast<RankedTensorType>(inputs[0].getType());

    // 1-D input case.
    if (inputType.getRank() == 1) {
      auto res = lower1DInput(inputs, op, rewriter);
      rewriter.replaceOp(op, res);
      return success();
    }

    // TODO: Optimization: The last two dimensions' data can be read with column
    // stride and computed in parallel.
    uint32_t axis = getAxis(op);
    assert(axis == (inputType.getRank() - 1) &&
           "Expected reduction axis is the last one");
    SmallVector<Value> res =
        lowering(inputs, op, rewriter, axis,
                 &ReduceScanOpConversionBase<OpT, ReturnOpT>::lower1DInput);

    rewriter.replaceOp(op, res);
    return success();
  }

  // In this case we either call lowerLeadingDimension to process the input
  // or extract sub-vectors, call lowerLeadingDimension, and then reconstruct
  // the result.
  LogicalResult
  lowerNonTrailingDimension(OpT op, ConversionPatternRewriter &rewriter) const {

    SmallVector<Value> inputs;
    if (failed(rewriter.getRemappedValues(getInputs(op), inputs)))
      return failure();

    uint32_t axis = getAxis(op);
    if (axis == 0) {
      rewriter.replaceOp(op, lowerLeadingDimension(inputs, op, rewriter));
      return success();
    }

    SmallVector<Value> res = lowering(
        inputs, op, rewriter, axis,
        &ReduceScanOpConversionBase<OpT, ReturnOpT>::lowerLeadingDimension);

    rewriter.replaceOp(op, res);
    return success();
  }

  // Accumulate inputs and existing accumulators into a new accumulators
  // applying operations from the combine region.
  SmallVector<Value> accumulate(ValueRange inputs, ValueRange acc,
                                Region &combineOp, OpBuilder &rewriter) const {
    if (acc.empty())
      return inputs;

    auto type = inputs[0].getType();
    SmallVector<int64_t> shape;
    if (isa<RankedTensorType>(type)) {
      auto temp = cast<RankedTensorType>(type).getShape();
      shape.insert(shape.end(), temp.begin(), temp.end());
    } else {
      shape.push_back(1);
    }
    auto &block = combineOp.getBlocks().front();
    IRMapping map;
    // Map block arguments to the current inputs and accumulators.
    for (unsigned i = 0; i < acc.size(); ++i) {
      map.map(block.getArgument(i), acc[i]);
      map.map(block.getArgument(acc.size() + i), inputs[i]);
    }
    for (auto &op : block.getOperations()) {
      // Returned values are a new accumulator.
      if (isa<ReturnOpT>(op)) {
        SmallVector<Value> res;
        for (auto operand : op.getOperands()) {
          res.push_back(map.lookup(operand));
        }
        return res;
      }

      // Clone operation mapping its inputs and building vector
      // result types using the input shape.
      OperationState newState(op.getLoc(), op.getName());
      for (auto operand : op.getOperands()) {
        newState.operands.push_back(
            lookupMappedValue(map, operand, shape, rewriter));
      }
      for (auto ty : op.getResultTypes()) {
        isa<RankedTensorType>(type)
            ? newState.types.push_back(RankedTensorType::get(shape, ty))
            : newState.types.push_back(ty);
      }
      newState.attributes = op.getAttrs();
      auto newOp = rewriter.create(newState);

      // Add new values to the map.
      for (auto [oldVal, newVal] :
           llvm::zip(op.getResults(), newOp->getResults())) {
        map.map(oldVal, newVal);
      }
    }
    llvm_unreachable("No return op found in scan/reduce region");
  }

  Value lookupMappedValue(IRMapping &localMap, Value val,
                          ArrayRef<int64_t> shape, OpBuilder &rewriter) const {

    Value res = localMap.lookupOrNull(val);
    if (!res) {
      // If value is not found then it's an invariant defined in the outer
      // region. We check if it has been already translated and add a splat
      // operation if it hasn't.
      res = invariantsMap.lookupOrNull(val);
      if (!res) {
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointAfterValue(val);
        res = rewriter.create<tensor::SplatOp>(
            val.getLoc(), RankedTensorType::get(shape, val.getType()), val);
        invariantsMap.map(val, res);
        rewriter.restoreInsertionPoint(ip);
      }
    }
    return res;
  }

  std::tuple<SmallVector<Value, 8>, SmallVector<int64_t>, SmallVector<int64_t>,
             SmallVector<ReassociationIndices>>
  tensorTransform(OpBuilder &rewriter, Location loc,
                  SmallVector<Value, 8> loopIndices,
                  RankedTensorType tensorType, uint32_t axis) const {
    auto shape = tensorType.getShape();

    SmallVector<Value, 8> dynamicIndices = loopIndices;
    dynamicIndices.insert(dynamicIndices.end(), shape.size() - axis,
                          rewriter.create<arith::ConstantIndexOp>(loc, 0));

    SmallVector<int64_t> staticSize(axis, 1);
    staticSize.insert(staticSize.end(), shape.begin() + axis, shape.end());
    SmallVector<int64_t> staticStride(shape.size(), 1);

    // {1,1,..(shape[axis])?,shape[axis+1],..shape[rank]}
    SmallVector<int64_t> extractShape = staticSize;

    // {1,1,..(shape[axis])?,shape[axis+1],..shape[rank]} ->
    // {(shape[axis])?,shape[axis+1],..shape[rank]}
    auto reassociationRank = shape.size() - axis;
    SmallVector<ReassociationIndices> reassociation(reassociationRank);
    if (reassociationRank) {
      // The first group: [0, 1, ..., axis - 1]
      reassociation[0].resize(axis);
      std::iota(reassociation[0].begin(), reassociation[0].end(), 0);
      // The remaining groups: [axis, axis+1, axis+2, ..., shape.size()-1]
      for (size_t i = axis; i < shape.size(); ++i) {
        reassociation[i - axis].push_back(i);
      }
    }

    return std::tuple<SmallVector<Value, 8>, SmallVector<int64_t>,
                      SmallVector<int64_t>, SmallVector<ReassociationIndices>>(
        dynamicIndices, staticSize, staticStride, reassociation);
  }

  using LoweringFuncType =
      SmallVector<Value> (ReduceScanOpConversionBase<OpT, ReturnOpT>::*)(
          ValueRange inputs, OpT op, ConversionPatternRewriter &rewriter) const;

  // Though function ptr to call lower1DInput/lowerLeadingDimension to handle
  // the tensor.
  SmallVector<Value> lowering(ValueRange inputs, OpT op,
                              ConversionPatternRewriter &rewriter,
                              uint32_t axis, LoweringFuncType handle) const {
    auto loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(inputs[0].getType());
    auto inputShape = inputType.getShape();

    auto outputType = cast<RankedTensorType>(op.getResults()[0].getType());
    auto outputShape = outputType.getShape();
    SmallVector<Value> res(inputs.size());
    std::transform(inputs.begin(), inputs.end(), res.begin(), [&](auto val) {
      return rewriter.create<tensor::EmptyOp>(loc, outputShape,
                                              inputType.getElementType());
    });

    SmallVector<scf::ForOp> loops;
    SmallVector<Value, 8> loopIndices;

    std::function<void(unsigned)> buildLoops = [&](unsigned d) {
      // Setup loop bounds and step.
      Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, inputShape[d]);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      SmallVector<Value> curRes =
          d == 0 ? res : loops[d - 1].getInitArgs().take_front(res.size());
      auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step,
                                              ValueRange{curRes});
      auto loopIdx = loop.getInductionVar();
      loopIndices.push_back(loopIdx);
      loops.push_back(loop);

      rewriter.setInsertionPointToStart(loop.getBody());
      if (d == axis - 1) {
        SmallVector<Value> subInputs(inputs.size());
        // [Dynamic indices, static size, static stride, reassociation]
        auto [inputDynamicIndices, inputStaticSize, inputStaticStride,
              inputReassociation] =
            tensorTransform(rewriter, loc, loopIndices, inputType, axis);
        for (size_t i = 0; i < inputs.size(); ++i) {
          auto extractTensor = rewriter.create<tensor::ExtractSliceOp>(
              loc,
              RankedTensorType::get(inputStaticSize,
                                    inputType.getElementType()),
              inputs[i], inputDynamicIndices, /*sizes*/ ValueRange(),
              /*strides*/ ValueRange(),
              SmallVector<int64_t>(inputShape.size(), ShapedType::kDynamic),
              inputStaticSize, inputStaticStride);
          subInputs[i] = rewriter.create<tensor::CollapseShapeOp>(
              loc, extractTensor, inputReassociation);
        }

        auto resElems = (this->*handle)(subInputs, op, rewriter);

        // [Dynamic indices, static size, static stride, reassociation]
        auto [outputDynamicIndices, outputStaticSize, outputStaticStride,
              outputReassociation] =
            tensorTransform(rewriter, loc, loopIndices, outputType, axis);

        for (size_t i = 0; i < res.size(); ++i) {
          auto targetType = RankedTensorType::get(outputStaticSize,
                                                  inputType.getElementType());
          // {shape[axis],shape[axis+1],..shape[rank]} ->
          // {1,1,..shape[axis],shape[axis+1],..shape[rank]}
          Value reshaped = rewriter.create<tensor::ExpandShapeOp>(
              loc, targetType, resElems[i], outputReassociation);

          curRes[i] = rewriter.create<tensor::InsertSliceOp>(
              loc, reshaped, curRes[i], outputDynamicIndices,
              /*sizes*/ ValueRange(),
              /*strides*/ ValueRange(),
              SmallVector<int64_t>(outputShape.size(), ShapedType::kDynamic),
              outputStaticSize, outputStaticStride);
        }
        rewriter.create<scf::YieldOp>(loc, curRes);
        return;
      }
      buildLoops(d + 1);
      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());
      SmallVector<Value> yieldValues =
          loops[d + 1].getResults().take_front(res.size());
      rewriter.create<scf::YieldOp>(loc, yieldValues);
      rewriter.setInsertionPointAfter(loop);
    };

    buildLoops(0);

    // Extract result tensors from forOp;
    SmallVector<Value> results;
    for (size_t i = 0; i < inputs.size(); ++i) {
      results.push_back(loops.front().getResult(i));
    }
    return results;
  }

private:
  mutable IRMapping invariantsMap;
};

TypedAttr getRedBaseAttr(OpBuilder &builder, Operation *redOp,
                         Type constantType);

arith::ConstantOp getRedBaseConstOp(ConversionPatternRewriter &rewriter,
                                    Operation *redOp, Type constantType);

} // namespace triton
} // namespace mlir
