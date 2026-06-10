#include "evas/Dialect/Linalg/IR/LinalgOpsExt.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::bufferization;
using namespace mlir::linalg;

namespace {

static OpFoldResult getDimValue(OpBuilder &builder, Location loc, Value value,
                                int64_t dim) {
  auto type = cast<ShapedType>(value.getType());
  if (!type.isDynamicDim(dim))
    return builder.getIndexAttr(type.getDimSize(dim));

  return getAsOpFoldResult(
      TypeSwitch<Type, Value>(value.getType())
          .Case<RankedTensorType>([&](RankedTensorType) -> Value {
            return builder.create<tensor::DimOp>(loc, value, dim);
          })
          .Case<MemRefType>([&](MemRefType) -> Value {
            return builder.create<memref::DimOp>(loc, value, dim);
          }));
}

static Operation *getSlice(OpBuilder &builder, Location loc, Value source,
                           ArrayRef<OpFoldResult> offsets,
                           ArrayRef<OpFoldResult> sizes,
                           ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Operation *>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType) -> Operation * {
        return builder.create<tensor::ExtractSliceOp>(loc, source, offsets,
                                                      sizes, strides);
      })
      .Case<MemRefType>([&](MemRefType) -> Operation * {
        return builder.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                                 strides);
      })
      .Default([](Type) { return nullptr; });
}

static void getCastEffects(
    CastOp op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (isa<MemRefType>(op.getInput().getType()))
    effects.emplace_back(MemoryEffects::Read::get(),
                         &op->getOpOperand(CastOp::odsIndex_input),
                         /*stage=*/0, /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());

  if (isa<MemRefType>(op.getOutput().getType())) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         &op->getOpOperand(CastOp::odsIndex_output),
                         /*stage=*/0, /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         &op->getOpOperand(CastOp::odsIndex_output),
                         /*stage=*/0, /*effectOnFullRegion=*/true,
                         SideEffects::DefaultResource::get());
  }
}

template <typename LinalgOp>
static LogicalResult bufferizeDestinationStyleOp(
    LinalgOp op, RewriterBase &rewriter, const BufferizationOptions &options,
    BufferizationState &state) {
  SmallVector<Value> operands;
  operands.reserve(op->getNumOperands());

  for (Value operand : op->getOperands()) {
    if (!isa<TensorType>(operand.getType())) {
      operands.push_back(operand);
      continue;
    }

    FailureOr<Value> buffer = getBuffer(rewriter, operand, options, state);
    if (failed(buffer))
      return failure();
    operands.push_back(*buffer);
  }

  auto newOp =
      rewriter.create<LinalgOp>(op.getLoc(), TypeRange(), operands,
                                op->getAttrs());
  auto dstOp = cast<DestinationStyleOpInterface>(newOp.getOperation());
  replaceOpWithBufferizedValues(rewriter, op.getOperation(),
                                dstOp.getDpsInits());
  return success();
}

} // namespace

void mlir::linalg::registerEvasLinalgOps(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *, linalg::LinalgDialect *dialect) {
    RegisteredOperationName::insert<linalg::CastOp>(*dialect);
  });
}

LogicalResult CastOp::verify() {
  ShapedType inputType = getInputOperandType();
  ShapedType outputType = getOutputOperandType();
  if (inputType.getRank() != outputType.getRank())
    return emitOpError("incompatible shape rank");

  for (auto [inputDim, outputDim] :
       llvm::zip_equal(inputType.getShape(), outputType.getShape())) {
    if (inputDim != outputDim)
      return emitOpError("wrong shape");
  }
  return success();
}

SmallVector<Range> CastOp::getIterationDomain(OpBuilder &builder) {
  int64_t rank = getInputOperandRank();
  SmallVector<Range> loopBounds(rank);
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value source = getInput();
  for (int64_t dim = 0; dim < rank; ++dim) {
    loopBounds[dim].offset = zero;
    loopBounds[dim].size = getDimValue(builder, loc, source, dim);
    loopBounds[dim].stride = one;
  }
  return loopBounds;
}

SmallVector<utils::IteratorType> CastOp::getLoopIteratorTypes() {
  return SmallVector<utils::IteratorType>(getInputOperandRank(),
                                          utils::IteratorType::parallel);
}

FailureOr<TilingResult>
CastOp::getTiledImplementation(OpBuilder &builder,
                               ArrayRef<OpFoldResult> offsets,
                               ArrayRef<OpFoldResult> sizes) {
  int64_t rank = getInputOperandRank();
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));

  Operation *inputSlice =
      getSlice(builder, getLoc(), getInput(), offsets, sizes, strides);
  if (!inputSlice)
    return emitOpError("failed to compute input slice");

  Operation *outputSlice =
      getSlice(builder, getLoc(), getOutput(), offsets, sizes, strides);
  if (!outputSlice)
    return emitOpError("failed to compute output slice");

  SmallVector<Value> tiledOperands{inputSlice->getResult(0),
                                  outputSlice->getResult(0)};
  SmallVector<Type> resultTypes;
  if (hasPureTensorSemantics())
    resultTypes.push_back(tiledOperands[1].getType());

  Operation *tiledOp =
      mlir::clone(builder, getOperation(), resultTypes, tiledOperands);

  return TilingResult{
      {tiledOp},
      SmallVector<Value>(tiledOp->getResults()),
      llvm::to_vector(ArrayRef<Operation *>{inputSlice, outputSlice})};
}

LogicalResult CastOp::getResultTilePosition(
    OpBuilder &, unsigned resultNumber, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, SmallVector<OpFoldResult> &resultOffsets,
    SmallVector<OpFoldResult> &resultSizes) {
  if (resultNumber != 0)
    return failure();
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return success();
}

FailureOr<TilingResult>
CastOp::generateResultTileValue(OpBuilder &builder, unsigned resultNumber,
                                ArrayRef<OpFoldResult> offsets,
                                ArrayRef<OpFoldResult> sizes) {
  SmallVector<OpFoldResult> resultOffsets;
  SmallVector<OpFoldResult> resultSizes;
  if (failed(getResultTilePosition(builder, resultNumber, offsets, sizes,
                                   resultOffsets, resultSizes)))
    return failure();
  return getTiledImplementation(builder, resultOffsets, resultSizes);
}

LogicalResult CastOp::fold(FoldAdaptor,
                           SmallVectorImpl<OpFoldResult> &results) {
  if (hasPureTensorSemantics()) {
    if (getInputOperandType() != getOutputOperandType())
      return failure();

    auto isInvalid = [](Value value) {
      Operation *defOp = value.getDefiningOp();
      if (!defOp)
        return true;
      return isa<tensor::ExtractSliceOp, tensor::ReshapeOp,
                 tensor::InsertSliceOp, tensor::ParallelInsertSliceOp,
                 tensor::ExpandShapeOp, tensor::CollapseShapeOp, tensor::CastOp,
                 bufferization::ToTensorOp>(defOp);
    };
    if (isInvalid(getInput()) || isInvalid(getOutput()))
      return failure();

    results.push_back(getInput());
    return success();
  }

  bool folded = false;
  for (OpOperand &operand : getOperation()->getOpOperands()) {
    auto cast = operand.get().getDefiningOp<memref::CastOp>();
    if (cast && !isa<UnrankedMemRefType>(cast.getOperand().getType()) &&
        !cast->hasAttr("no_fold")) {
      operand.set(cast.getOperand());
      folded = true;
    }
  }
  return success(folded);
}

void CastOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  getCastEffects(*this, effects);
}

bool CastOp::bufferizesToMemoryRead(OpOperand &,
                                    const AnalysisState &) {
  return true;
}

bool CastOp::bufferizesToMemoryWrite(OpOperand &opOperand,
                                     const AnalysisState &) {
  return opOperand.getOperandNumber() == CastOp::odsIndex_output;
}

AliasingValueList CastOp::getAliasingValues(OpOperand &opOperand,
                                            const AnalysisState &) {
  if (opOperand.getOperandNumber() == CastOp::odsIndex_output)
    return {{getOperation()->getResult(0), BufferRelation::Equivalent}};
  return {};
}

LogicalResult CastOp::bufferize(RewriterBase &rewriter,
                                const BufferizationOptions &options,
                                BufferizationState &state) {
  return bufferizeDestinationStyleOp(*this, rewriter, options, state);
}

#include "evas/Dialect/Linalg/IR/LinalgOpsExtEnums.cpp.inc"

#define GET_OP_CLASSES
#include "evas/Dialect/Linalg/IR/LinalgOpsExt.cpp.inc"
