//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ConvertScanOp.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include <numeric>

using namespace mlir;
using namespace triton;

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTSCANOP
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
} // namespace mlir::triton

namespace {

// Base class for converting scans and reductions.
//
// It provides accumulation function that clones operations from the
// original combine region and applies them on provided vectors.
// Also, it handles multi-diumensional cases reducing them to two
// possible options: lowering for a 1-D vector inputs and lowering
// the operation over the leading dimension.
//
// Specialized pattern should implement lower1DInput to handle
// trailing dimension case (commonly through shuffles + accumulate)
// and lowerLeadingDimension to handle the leading dimension case
// through accumulation of sub-vectors.
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

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rank = cast<RankedTensorType>(op.getOperand(0).getType()).getRank();
    if (op.getAxis() == (rank - 1))
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
    if (failed(rewriter.getRemappedValues(op.getOperands(), inputs)))
      return failure();

    SmallVector<VectorType> inputTys(inputs.size());
    std::transform(inputs.begin(), inputs.end(), inputTys.begin(),
                   [](auto val) { return cast<VectorType>(val.getType()); });

    // 1-D input case.
    if (inputTys.front().getRank() == 1) {
      auto res = lower1DInput(inputs, op, rewriter);
      rewriter.replaceOp(op, res);
      return success();
    }

    SmallVector<Value> res =
        makeEmptyResults(loc, op.getResultTypes(), rewriter);
    auto shape = inputTys[0].getShape();
    int64_t numElems = inputTys[0].getNumElements();
    auto strides = computeStrides(shape);
    // Remove the last stride to produce sub-vector indices.
    strides.pop_back();
    for (int64_t idx = 0; idx < numElems; idx += shape.back()) {
      auto indices = delinearize(idx, strides);
      SmallVector<Value> subInputs(inputs.size());
      std::transform(
          inputs.begin(), inputs.end(), subInputs.begin(), [&](auto val) {
            return vector::ExtractOp::create(rewriter, loc, val, indices);
          });

      auto resElems = lower1DInput(subInputs, op, rewriter);
      for (size_t i = 0; i < res.size(); ++i) {
        res[i] = vector::InsertOp::create(rewriter, loc, resElems[i], res[i],
                                          indices);
      }
    }

    rewriter.replaceOp(op, res);
    return success();
  }

  // In this case we either call lowerLeadingDimension to process the input
  // or extract sub-vectors, call lowerLeadingDimension, and then reconstruct
  // the result.
  LogicalResult
  lowerNonTrailingDimension(OpT op, ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    SmallVector<Value> inputs;
    if (failed(rewriter.getRemappedValues(op.getOperands(), inputs)))
      return failure();

    uint32_t axis = op.getAxis();
    if (axis == 0) {
      rewriter.replaceOp(op, lowerLeadingDimension(inputs, op, rewriter));
      return success();
    }

    SmallVector<Value> res =
        makeEmptyResults(loc, op.getResultTypes(), rewriter);
    auto vecTy = cast<VectorType>(inputs[0].getType());
    auto shape = vecTy.getShape();
    auto strides = computeStrides(shape);
    // Remove trailing elems to build indices of required rank.
    strides.erase(strides.begin() + axis, strides.end());
    int64_t numElems = vecTy.getNumElements();
    int64_t step = strides.back();
    for (int64_t idx = 0; idx < numElems; idx += step) {
      auto indices = delinearize(idx, strides);
      SmallVector<Value> subInputs(inputs.size());
      std::transform(
          inputs.begin(), inputs.end(), subInputs.begin(), [&](auto val) {
            return vector::ExtractOp::create(rewriter, loc, val, indices);
          });
      auto resVecs = lowerLeadingDimension(subInputs, op, rewriter);
      for (size_t i = 0; i < res.size(); ++i) {
        res[i] = vector::InsertOp::create(rewriter, loc, resVecs[i], res[i],
                                          indices);
      }
    }

    rewriter.replaceOp(op, res);
    return success();
  }

  // Accumulate inputs and existing accumulators into a new accumaltors
  // applying operations from the combine region.
  SmallVector<Value> accumulate(ValueRange inputs, ValueRange acc,
                                Region &combineOp,
                                ConversionPatternRewriter &rewriter) const {
    if (acc.empty())
      return inputs;

    auto shape = cast<VectorType>(inputs[0].getType()).getShape();
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
        newState.types.push_back(VectorType::get(shape, ty));
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
                          ArrayRef<int64_t> shape,
                          ConversionPatternRewriter &rewriter) const {

    Value res = localMap.lookupOrNull(val);
    if (!res) {
      // If value is not found then it's an invariant defined in the outer
      // region. We check if it has been already translated and add a splat
      // operation if it hasn't.
      res = invariantsMap.lookupOrNull(val);
      if (!res) {
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointAfterValue(val);
        res = vector::BroadcastOp::create(
            rewriter, val.getLoc(), VectorType::get(shape, val.getType()), val);
        invariantsMap.map(val, res);
        rewriter.restoreInsertionPoint(ip);
      }
    }
    return res;
  }

  SmallVector<Value>
  makeEmptyResults(Location loc, TypeRange resTypes,
                   ConversionPatternRewriter &rewriter) const {
    // Initialize results to zero values.
    SmallVector<Value> res;
    for (auto ty : resTypes) {
      res.push_back(arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getZeroAttr(getTypeConverter()->convertType(ty))));
    }
    return res;
  }

  // Dummy vectors are required for shuffles that cannot work on a single
  // vector.
  SmallVector<Value>
  createShuffleDummies(Location loc, ValueRange inputs,
                       ConversionPatternRewriter &rewriter) const {
    SmallVector<Value> shuffleDummies;
    SmallVector<int64_t, 1> dummyShape({1});
    for (auto val : inputs) {
      auto ty = cast<VectorType>(val.getType());
      shuffleDummies.push_back(arith::ConstantOp::create(
          rewriter, loc,
          rewriter.getZeroAttr(ty.cloneWith(dummyShape, ty.getElementType()))));
    }
    return shuffleDummies;
  }

private:
  mutable IRMapping invariantsMap;
};

class TritonToTritonLinalgTypeConverter : public TypeConverter {
public:
  using TypeConverter::convertType;

  TritonToTritonLinalgTypeConverter();

  Type convertTritonPointerType(triton::PointerType type);
};

TritonToTritonLinalgTypeConverter::TritonToTritonLinalgTypeConverter() {
  addConversion([](Type type) { return type; });
  addConversion([this](RankedTensorType tensorTy) -> Type {
    Type elemTy = convertType(tensorTy.getElementType());
    if (isa<triton::PointerType>(elemTy))
      elemTy = IntegerType::get(tensorTy.getContext(), 64);
    return VectorType::get(tensorTy.getShape(), elemTy);
  });

  // Converted ops produce vectors instead of tensors. Provide conversion
  // here for users.
  addSourceMaterialization([&](OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) -> mlir::Value {
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  });

  // Provide conversion for vector users.
  addTargetMaterialization([&](OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) -> mlir::Value {
    if (isa<VectorType>(type))
      return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
          .getResult(0);
    llvm_unreachable("Unexpected target materizalization");
  });
}

class ScanConversionTarget : public ConversionTarget {
public:
  explicit ScanConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {

    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<linalg::LinalgDialect>();
    addLegalDialect<func::FuncDialect>();
    addLegalDialect<math::MathDialect>();
    addLegalDialect<affine::AffineDialect>();
    addLegalDialect<bufferization::BufferizationDialect>();
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<memref::MemRefDialect>();
    addLegalDialect<scf::SCFDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addLegalDialect<tensor::TensorDialect>();

    // addLegalOp<mlir::UnrealizedConversionCastOp>();

    addIllegalOp<triton::ScanOp>();
  }
};

struct ScanOpConversion
    : public ReduceScanOpConversionBase<triton::ScanOp, triton::ScanReturnOp> {
  using ReduceScanOpConversionBase::ReduceScanOpConversionBase;

  SmallVector<Value>
  lower1DInput(ValueRange inputs, ScanOp op,
               ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Region &combineOp = op.getRegion();
    bool reverse = op.getReverse();
    int64_t vecSize = cast<VectorType>(inputs[0].getType()).getShape()[0];
    Type maskTy = VectorType::get(vecSize, rewriter.getI1Type());

    SmallVector<Value> dummies = createShuffleDummies(loc, inputs, rewriter);
    SmallVector<Value> res = inputs;
    for (int64_t stride = 1; stride < vecSize; stride *= 2) {
      SmallVector<int64_t> shuffleIndices(vecSize, 0);
      int64_t start = reverse ? vecSize - 1 - stride : stride;
      int64_t end = reverse ? -1 : vecSize;
      int64_t step = reverse ? -1 : 1;
      for (int64_t i = start; i != end; i += step) {
        shuffleIndices[i] = i - step * stride;
      }
      SmallVector<Value> shuffledInput;
      for (auto [val, dummy] : llvm::zip(res, dummies)) {
        shuffledInput.push_back(vector::ShuffleOp::create(
            rewriter, loc, val, dummy, shuffleIndices));
      }

      auto newRes = accumulate(res, shuffledInput, combineOp, rewriter);

      // Number of already computed elements is equal to the current
      // stride. Mask them out using a constant mask.
      SmallVector<bool> maskVals(vecSize, true);
      if (reverse) {
        std::fill(maskVals.rbegin(), maskVals.rbegin() + stride, false);
      } else {
        std::fill(maskVals.begin(), maskVals.begin() + stride, false);
      }
      Value mask = arith::ConstantOp::create(
          rewriter, loc, maskTy, rewriter.getBoolVectorAttr(maskVals));
      for (size_t i = 0; i < res.size(); ++i) {
        res[i] = vector::selectPassthru(rewriter, mask, newRes[i], res[i]);
      }
    }

    return res;
  }

  SmallVector<Value>
  lowerLeadingDimension(ValueRange inputs, ScanOp op,
                        ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Region &combineOp = op.getRegion();
    bool reverse = op.getReverse();
    auto shape = cast<VectorType>(inputs[0].getType()).getShape();
    SmallVector<Type> resTypes;
    for (const auto &resTy : op.getResultTypes()) {
      resTypes.push_back(VectorType::get(
          shape, cast<RankedTensorType>(resTy).getElementType()));
    }
    SmallVector<Value> res = makeEmptyResults(loc, resTypes, rewriter);
    SmallVector<Value> acc;
    int64_t start = reverse ? shape[0] - 1 : 0;
    int64_t end = reverse ? -1 : shape[0];
    int64_t step = reverse ? -1 : 1;
    for (int64_t idx = start; idx != end; idx += step) {
      SmallVector<Value> subInputs(inputs.size());
      std::transform(
          inputs.begin(), inputs.end(), subInputs.begin(), [&](auto val) {
            return vector::ExtractOp::create(rewriter, loc, val, idx);
          });

      acc = accumulate(subInputs, acc, combineOp, rewriter);

      for (size_t i = 0; i < res.size(); ++i) {
        res[i] = vector::InsertOp::create(rewriter, loc, acc[i], res[i], idx);
      }
    }
    return res;
  }
};

class TensorToVectorCastConverter
    : public OpRewritePattern<UnrealizedConversionCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Value input = op->getOperand(0);
    Value result = op.getResult(0);

    auto tensorType = dyn_cast<TensorType>(input.getType());
    auto vectorType = dyn_cast<VectorType>(result.getType());
    if (!tensorType || !vectorType)
      return failure();

    if (tensorType.getElementType() != vectorType.getElementType() ||
        tensorType.getShape() != vectorType.getShape())
      return failure();

    Location loc = op.getLoc();
    Value memref;

    Operation *definingOp = input.getDefiningOp();
    if (definingOp && isa<bufferization::ToTensorOp>(definingOp)) {
      memref = definingOp->getOperand(0);
    } else {
      auto memrefType =
          MemRefType::get(tensorType.getShape(), tensorType.getElementType());
      memref =
          bufferization::ToBufferOp::create(rewriter, loc, memrefType, input);
    }

    SmallVector<Value> indices;
    auto shape = tensorType.getShape();
    indices.reserve(shape.size());
    for (int64_t dim : shape) {
      (void)dim;
      indices.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));
    }

    Attribute zeroAttr = rewriter.getZeroAttr(tensorType.getElementType());
    Value padding = arith::ConstantOp::create(
        rewriter, loc, tensorType.getElementType(), cast<TypedAttr>(zeroAttr));

    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(op, vectorType, memref,
                                                        indices, padding);

    return success();
  }
};

class VectorToTensorCastConverter
    : public OpRewritePattern<UnrealizedConversionCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();

    Value input = op->getOperand(0);
    Value ouput = op.getResult(0);
    auto vectorType = dyn_cast<VectorType>(input.getType());
    auto tensorType = dyn_cast<TensorType>(ouput.getType());
    if (!vectorType || !tensorType)
      return failure();

    if (vectorType.getElementType() != tensorType.getElementType() ||
        vectorType.getShape() != tensorType.getShape())
      return failure();

    Location loc = op.getLoc();
    MemRefType memRefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType());
    Value alloc = memref::AllocOp::create(rewriter, loc, memRefType);

    SmallVector<Value> indices;
    for (int64_t dim : vectorType.getShape()) {
      (void)dim;
      indices.push_back(arith::ConstantIndexOp::create(rewriter, loc, 0));
    }

    vector::TransferWriteOp::create(rewriter, loc, input, alloc,
                                    ValueRange{indices});

    auto toTensor =
        bufferization::ToTensorOp::create(rewriter, loc, tensorType, alloc,
                                          /*restrict=*/true, /*writable=*/true);

    rewriter.replaceOp(op, toTensor.getResult());
    return success();
  }
};

struct ConvertScanOpPass
    : public triton::impl::ConvertScanOpBase<ConvertScanOpPass> {
  using ConvertScanOpBase::ConvertScanOpBase;

  ConvertScanOpPass() : ConvertScanOpBase() {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, bufferization::BufferizationDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonLinalgTypeConverter typeConverter;
    ScanConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<ScanOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    RewritePatternSet patternsCast(&getContext());
    patternsCast.add<TensorToVectorCastConverter, VectorToTensorCastConverter>(
        &getContext());
    if (failed(
            applyPartialConversion(mod, convTarget, std::move(patternsCast))))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createConvertScanOpPass() {
  return std::make_unique<ConvertScanOpPass>();
}
