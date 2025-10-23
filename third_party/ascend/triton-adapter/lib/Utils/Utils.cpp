//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#define DEBUG_TYPE "TritonNPU-Utils"

namespace mlir {

static Value createConstIndexValueOp(const Location &loc, OpBuilder &b,
                                     int64_t value) {
  return b.create<arith::ConstantOp>(loc, b.getIndexAttr(value)).getResult();
}

static std::optional<int64_t> getConstantOfAttr(const OpFoldResult &arg) {
  if (isa<Attribute>(arg)) {
    return getConstantIntValue(arg);
  }

  return std::nullopt;
}

namespace ConverterUtils {

std::optional<int64_t> getLastStrideOfReinterpretCastOp(memref::ReinterpretCastOp op) {
  SmallVector<OpFoldResult> mixedStrides = op.getMixedStrides();
  if (mixedStrides.empty()) {
      op->emitError("ReinterpretCastOp has no strides");
      return std::nullopt;
  }

  OpFoldResult lastStride = mixedStrides.back();
  if (auto attr = lastStride.dyn_cast<Attribute>()) {
    return getConstantOfAttr(lastStride);
  } else if (auto value = lastStride.dyn_cast<Value>()) {
    auto defOp = value.getDefiningOp();
    if (auto constIndexOp = dyn_cast<arith::ConstantIndexOp>(defOp)) {
      int64_t constValue = constIndexOp.value();
      return constValue;
    }
    else if (auto constIntOp = dyn_cast<arith::ConstantIntOp>(defOp)) {
      int64_t constValue = constIntOp.value();
      return constValue;
    }
  } 
  return std::nullopt;
}

bool isaPermutedMemRefType(MemRefType memRefType) {
  auto [ptrStrides, ptrOffsets] = getStridesAndOffset(memRefType);
  LLVM_DEBUG({
    llvm::dbgs()<<"---------- [BEG] ptrStrides ----------\n";
    for(auto stride: ptrStrides)llvm::dbgs()<<stride<<" ";llvm::dbgs()<<"\n";
    llvm::dbgs()<<"---------- [END] ptrStrides ----------\n";
  });

  switch (ptrStrides.size()) {
    case 0: return false;
    case 1: return false;
    default: {
      return ptrStrides[ptrStrides.size()-1] != 1;
    }
  }
}

Value getTransposedValue(Value source, const Location loc,
                         ConversionPatternRewriter &rewriter,
                         llvm::ArrayRef<int> order) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(order);
  SmallVector<int64_t> originalShape(sourceType.getShape());
  SmallVector<int64_t> transposedShape(sourceRank);
  for (size_t i = 0; i < sourceRank; i++) {
    transposedShape[i] = originalShape[perm[i]];
  }

  Value transposeInit = rewriter.create<tensor::EmptyOp>(
      loc, transposedShape, sourceType.getElementType());

  Value transpose =
      rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
          .getResults()[0];

  return transpose;
}

SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

Value getScalarValue(Value operand, Location loc,
                     ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *> ops;
  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = mlir::TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::SIToFPOp>(loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::TruncFOp>(loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitError(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            rewriter, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

memref::SubViewOp makeSubViewOp(Value src,
                                const llvm::SmallVector<OpFoldResult> &sizes,
                                const Location &loc,
                                ConversionPatternRewriter &rewriter) {
  auto srcType = cast<MemRefType>(src.getType());
  SmallVector<OpFoldResult> offsets(srcType.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(srcType.getRank(),
                                    rewriter.getIndexAttr(1));
  auto dstType =
      memref::SubViewOp::inferResultType(srcType, offsets, sizes, strides);
  return rewriter.create<memref::SubViewOp>(loc, dyn_cast<MemRefType>(dstType),
                                            src, offsets, sizes, strides);
}

tensor::ExtractSliceOp makeExtractSliceOp(Value src,
                                          const llvm::SmallVector<OpFoldResult> &sizes,
                                          const Location &loc,
                                          ConversionPatternRewriter &rewriter) {
  auto srcType = cast<RankedTensorType>(src.getType());
  SmallVector<OpFoldResult> offsets(srcType.getRank(),
                                    rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(srcType.getRank(),
                                    rewriter.getIndexAttr(1));
  auto dstType =
      tensor::ExtractSliceOp::inferResultType(srcType, offsets, sizes, strides);
  return rewriter.create<tensor::ExtractSliceOp>(loc, dstType, src, offsets,
                                                 sizes, strides);
}

std::optional<Operation *> getFullShapeOp(Value val,
                                          ConversionPatternRewriter &rewriter) {
  assert(isa<BaseMemRefType>(val.getType()));

  if (isa<BlockArgument>(val)) {
    auto blockArg = dyn_cast<BlockArgument>(val);
    auto blockOp = blockArg.getOwner()->getParentOp();
    if (isa<scf::ForOp>(blockOp)) {
      auto forOp = dyn_cast<scf::ForOp>(blockOp);
      auto operand = forOp.getTiedLoopInit(blockArg)->get();
      return getFullShapeOp(operand, rewriter);
    } else {
      emitError(val.getLoc())
          << "getFullShapeOp() only support ReinterpretCastOp "
             "and scf.for's block argument, but got : "
          << val << "\n";
    }
    return std::nullopt;
  }

  if (!isa<memref::ReinterpretCastOp>(val.getDefiningOp())) {
    emitError(val.getLoc())
        << "getFullShapeOp() only support ReinterpretCastOp "
           "and scf.for's block argument, but got : "
        << val << "\n";
    return std::nullopt;
  }

  auto reCastOp = val.getDefiningOp<memref::ReinterpretCastOp>();
  if (reCastOp->hasAttr("tensor_ptr_full_shape"))
    return reCastOp;

  return getFullShapeOp(reCastOp.getSource(), rewriter);
}

SmallVector<OpFoldResult>
getBoundarySizes(llvm::ArrayRef<int32_t> boundaryCheck, Value ptr,
                 const Location &loc, ConversionPatternRewriter &rewriter) {
  if (isa<triton::PointerType>(ptr.getType()))
    ptr = rewriter.getRemappedValue(ptr);

  auto shapedType = dyn_cast_if_present<ShapedType>(ptr.getType());
  assert(shapedType && shapedType.hasStaticShape());

  auto fullShapeOp = getFullShapeOp(ptr, rewriter);

  assert(fullShapeOp.has_value());
  SmallVector<OpFoldResult> boundarySize =
      getAsIndexOpFoldResult(rewriter.getContext(), shapedType.getShape());

  auto fullShapeReCast =
      dyn_cast<memref::ReinterpretCastOp>(fullShapeOp.value());
  OpFoldResult curPtrOffset;
  if (auto curReCast = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
    curPtrOffset = curReCast.getConstifiedMixedOffset();
  } else if (isa<BlockArgument>(ptr) &&
             isa<scf::ForOp>(ptr.getParentBlock()->getParentOp())) {
    // Here's to process loop state where ptr is just from loop interator.
    // Following assertion corresponds to conversion result from `rewriteFor`
    auto blockArg = dyn_cast<BlockArgument>(ptr);
    auto forOp = dyn_cast<scf::ForOp>(ptr.getParentBlock()->getParentOp());
    auto initReCastOfLoop = forOp.getTiedLoopInit(blockArg)
                                ->get()
                                .getDefiningOp<memref::ReinterpretCastOp>();
    assert(initReCastOfLoop && initReCastOfLoop.getOffsets().size() == 1);
    Value initReCastOffset = initReCastOfLoop.getOffsets()[0];

    for (OpOperand &use : initReCastOffset.getUses()) {
      if (use.getOwner() == initReCastOfLoop)
        continue;
      else if (isa<memref::ReinterpretCastOp>(use.getOwner()))
        continue;
      else if (use.getOwner() == forOp)
        curPtrOffset = OpFoldResult(forOp.getTiedLoopRegionIterArg(&use));
      else
        llvm_unreachable("Illegal interation offset after rewriteFor");
    }
  } else {
    llvm_unreachable("Unsupported state when check tensor_ptr boundary");
  }

  assert(curPtrOffset);

  OpFoldResult offsetShift = subOpFoldResult(
      curPtrOffset, fullShapeReCast.getConstifiedMixedOffset(), loc, rewriter);

  for (int i = 0; i < shapedType.getRank(); ++i) {
    if (llvm::find(boundaryCheck, i) != boundaryCheck.end()) {
      auto fullShape = fullShapeReCast.getConstifiedMixedSizes()[i];

      OpFoldResult curOffset = divOpFoldResult(
          offsetShift, fullShapeReCast.getConstifiedMixedStrides()[i], loc,
          rewriter);
      OpFoldResult curLeftSize =
          maxOpFoldResult(subOpFoldResult(fullShape, curOffset, loc, rewriter),
                          rewriter.getIndexAttr(0), loc, rewriter);

      boundarySize[i] =
          minOpFoldResult(boundarySize[i], curLeftSize, loc, rewriter);

      offsetShift = remOpFoldResult(
          offsetShift, fullShapeReCast.getConstifiedMixedStrides()[i], loc,
          rewriter);
    }
  }

  return boundarySize;
}

SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                      RankedTensorType dst) {
  SmallVector<int64_t> broadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] != srcShape[i]) {
      assert(srcShape[i] == 1 &&
             "Size of source broadcast dimension must be 1");
      broadcastDims.push_back(i);
    }
  }
  assert(!broadcastDims.empty() && "Cannot identify broadcast dimension");
  return broadcastDims;
}

// Dimensions of collapesd tensor is all unbroadcast dims
SmallVector<int64_t> getUnbroadcastDims(RankedTensorType src,
                                        RankedTensorType dst) {
  SmallVector<int64_t> unbroadcastDims;
  auto srcShape = src.getShape();
  auto dstShape = dst.getShape();

  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (dstShape[i] == srcShape[i]) {
      unbroadcastDims.emplace_back(srcShape[i]);
    }
  }
  return unbroadcastDims;
}

} // namespace ConverterUtils

namespace triton {

mlir::Operation *
findFirstMatchingOperandDef(mlir::Operation *rootOp,
                            const std::function<bool(Operation *)> &condFn) {
  LLVM_DEBUG(llvm::dbgs() << "[findFirstMatchingOperandDef] Current op: "
                          << *rootOp << "\n");
  mlir::Value lhs = nullptr;
  mlir::Value rhs = nullptr;
  if (auto op = dyn_cast<triton::AddPtrOp>(rootOp)) {
    lhs = op.getPtr();
    rhs = op.getOffset();
  } else if (auto op = dyn_cast<arith::AddIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::SubIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::MulIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::DivSIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<arith::RemSIOp>(rootOp)) {
    lhs = op.getLhs();
    rhs = op.getRhs();
  } else if (auto op = dyn_cast<triton::SplatOp>(rootOp)) {
    lhs = op.getSrc();
  } else if (auto op = dyn_cast<triton::MakeRangeOp>(rootOp)) {
  } else {
    rootOp->emitRemark("Backtracing encounters unsupported Operation");
    return nullptr;
  }
  // Backtrace operands
  if (!lhs) {
    return nullptr;
  }
  auto lhsDef = lhs.getDefiningOp();
  mlir::Operation *targetOp;
  if (lhsDef) {
    if (condFn(lhsDef)) {
      targetOp = lhsDef;
    } else {
      targetOp = findFirstMatchingOperandDef(lhsDef, condFn);
    }
    if (targetOp) {
      return targetOp;
    }
  }
  if (!rhs) {
    return nullptr;
  }
  auto rhsDef = rhs.getDefiningOp();
  if (rhsDef) {
    if (condFn(rhsDef)) {
      targetOp = rhsDef;
    } else {
      targetOp = findFirstMatchingOperandDef(rhsDef, condFn);
    }
    if (targetOp) {
      return targetOp;
    }
  }
  return nullptr;
}

void traverseBackwardUpdateOperandChainIf(
    Operation *op, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn, OpBuilder &builder,
    DenseSet<Operation *> &handledOperation) {

  if (!op || handledOperation.contains(op))
    return;

  handledOperation.insert(op);

  if (stopFn(op))
    return;

  if (conditionFn(op))
    actionFn(builder, op);

  DenseSet<Value> handledOperand;

  std::function<void(Value)> handler = [&](Value operand) {
    if (handledOperand.contains(operand))
      return;
    handledOperand.insert(operand);
    if (Operation *defOp = operand.getDefiningOp()) {
      traverseBackwardUpdateOperandChainIf(defOp, conditionFn, stopFn, actionFn,
                                           builder, handledOperation);
    } else {
      auto blockArgument = cast<BlockArgument>(operand);
      if (auto loopOp =
          dyn_cast<LoopLikeOpInterface>(blockArgument.getOwner()->getParentOp())) {
        OpOperand *initArgOperand = loopOp.getTiedLoopInit(blockArgument);
        if (!initArgOperand)
          return;
        Value initArg = initArgOperand->get();
        handler(initArg);
        Value yieldedValue = loopOp.getTiedLoopYieldedValue(blockArgument)->get();
        if (yieldedValue != blockArgument)
          handler(yieldedValue);
      }
    }
  };

  for (Value operand : op->getOperands()) {
    handler(operand);
  }

  if (auto loopOp = dyn_cast<LoopLikeOpInterface>(op)) {
    for (auto yieldedValue: loopOp.getYieldedValues())
      handler(yieldedValue);
  }
}

// Note: rootOp will also be processed.
void traverseBackwardUpdateOperandChainIf(
    Operation *rootOp, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn) {

  OpBuilder builder(rootOp->getContext());
  DenseSet<Operation *> handledOperation;

  traverseBackwardUpdateOperandChainIf(rootOp, conditionFn, stopFn, actionFn, builder,
                                       handledOperation);
}

void traverseForwardUpdateUserChainIf(
    Operation *op, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn, OpBuilder &builder,
    llvm::SmallPtrSet<Operation *, 16> &stopOps) {

  if (!op) {
    return;
  }

  if (stopFn(op)) {
    stopOps.insert(op);
    return;
  }

  if (conditionFn(op)) {
    actionFn(builder, op);
  }

  for (auto res : op->getResults()) {
    for (auto userOp : res.getUsers()) {
      traverseForwardUpdateUserChainIf(userOp, conditionFn, stopFn, actionFn,
                                       builder, stopOps);
    }
  }
}

// Note: rootOp will also be processed.
void traverseForwardUpdateUserChainIf(
    Operation *rootOp, std::function<bool(Operation *)> conditionFn,
    std::function<bool(Operation *)> stopFn,
    std::function<void(OpBuilder &, Operation *)> actionFn,
    llvm::SmallPtrSet<Operation *, 16> &stopOps) {

  OpBuilder builder(rootOp->getContext());

  traverseForwardUpdateUserChainIf(rootOp, conditionFn, stopFn, actionFn,
                                   builder, stopOps);
}

bool isMetaUse(Operation *op) { return op->hasAttr("MetaUse"); }

bool isMixUse(Operation *op) { return op->hasAttr("MixUse"); }

IndirectLoadInterfaceOpType getIndirectLoadInterfaceOpType(Operation *op) {
  auto ty = IndirectLoadInterfaceOpType::Undefined;
  if (isMetaUse(op)) {
    if (isa<triton::LoadOp>(op)) {
      ty = IndirectLoadInterfaceOpType::Load;
    } else if (isa<arith::FPToSIOp>(op)) {
      ty = IndirectLoadInterfaceOpType::Calc;
    }
  }
  return ty;
}

bool opIsIndirectLoad(Operation *op) {
  auto opType = getIndirectLoadInterfaceOpType(op);
  return opType == IndirectLoadInterfaceOpType::Load;
}

bool opIsIndirectCalc(Operation *op) {
  auto opType = getIndirectLoadInterfaceOpType(op);
  return opType == IndirectLoadInterfaceOpType::Calc;
}

scf::ForOp createNestedLoops(
    OpBuilder &builder, Location loc, unsigned currentDim, unsigned totalDims,
    ValueRange LBs, ValueRange UBs, ValueRange steps, SmallVector<Value> &ivs,
    ValueRange initArgs,
    function_ref<void(OpBuilder &, Location, SmallVector<Value> &, ValueRange)>
        bodyBuilder) {

  if (currentDim >= totalDims) {
    bodyBuilder(builder, loc, ivs, initArgs);
    return nullptr;
  }

  auto loop = builder.create<scf::ForOp>(
      loc, LBs[currentDim], UBs[currentDim], steps[currentDim], initArgs,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, Value iv,
          ValueRange iterArgs) {
        ivs.push_back(iv);
        auto innerLoop = createNestedLoops(nestedBuilder, nestedLoc,
                                           currentDim + 1, totalDims, LBs, UBs,
                                           steps, ivs, iterArgs, bodyBuilder);
        if (innerLoop) {
          nestedBuilder.create<scf::YieldOp>(loc, innerLoop.getResults());
        }
      });

  return loop;
}

ModuleOp getModuleOpFromOperation(Operation *op) {
  Operation *parent = op;
  while (parent != nullptr && !isa<ModuleOp>(parent)) {
    parent = parent->getParentOp(); // 向上查找
  }
  return cast<ModuleOp>(parent); // 如果没找到会抛出异常
}

} // namespace triton


// TODO: imply these function below
OpFoldResult addOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() + rhsInt.value());

  if (!lhsInt && rhsInt && rhsInt.value() == 0)
    return lhs;
  if (!rhsInt && lhsInt && lhsInt.value() == 0)
    return rhs;

  auto lhsValue = dyn_cast<Value>(lhs);
  if (lhsInt)
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  else
    assert(isa<IndexType>(lhsValue.getType()));

  auto rhsValue = dyn_cast<Value>(rhs);
  if (rhsInt)
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  else
    assert(isa<IndexType>(rhsValue.getType()));

  return b.create<arith::AddIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult subOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() - rhsInt.value());

  if (!lhsInt && rhsInt && rhsInt.value() == 0)
    return lhs;

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt)
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  else
    assert(isa<IndexType>(lhsValue.getType()));

  if (rhsInt)
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  else
    assert(isa<IndexType>(rhsValue.getType()));

  return b.create<arith::SubIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult mulOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() * rhsInt.value());

  if (lhsInt) {
    if (lhsInt.value() == 0)
      return lhs;
    if (lhsInt.value() == 1)
      return rhs;
  }
  if (rhsInt) {
    if (rhsInt.value() == 0)
      return rhs;
    if (rhsInt.value() == 1)
      return lhs;
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt)
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  else
    assert(isa<IndexType>(lhsValue.getType()));

  if (rhsInt)
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  else
    assert(isa<IndexType>(rhsValue.getType()));

  return b.create<arith::MulIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult divOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (rhsInt && rhsInt.value() == 0) {
    emitError(loc) << "cannot div 0!";
    return OpFoldResult();
  }

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() / rhsInt.value());

  if (lhsInt) {
    if (lhsInt.value() == 0)
      return lhs;
  }

  if (rhsInt) {
    if (rhsInt.value() == 1)
      return lhs;
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt)
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  else
    assert(isa<IndexType>(lhsValue.getType()));

  if (rhsInt)
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  else
    assert(isa<IndexType>(rhsValue.getType()));

  return b.create<arith::DivSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult remOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);

  if (rhsInt && rhsInt.value() == 0) {
    emitError(loc) << "cannot remainder by 0!";
    return OpFoldResult();
  }

  if (lhsInt && rhsInt)
    return b.getIndexAttr(lhsInt.value() % rhsInt.value());

  if (lhsInt) {
    if (lhsInt.value() == 0)
      return lhs;
  }

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt)
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  else
    assert(isa<IndexType>(lhsValue.getType()));

  if (rhsInt)
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  else
    assert(isa<IndexType>(rhsValue.getType()));

  return b.create<arith::RemSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult minOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);
  if (lhsInt && rhsInt)
    return b.getIndexAttr(std::min(lhsInt.value(), rhsInt.value()));

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt)
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  else
    assert(isa<IndexType>(lhsValue.getType()));

  if (rhsInt)
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  else
    assert(isa<IndexType>(rhsValue.getType()));

  return b.create<arith::MinSIOp>(loc, lhsValue, rhsValue).getResult();
}

OpFoldResult maxOpFoldResult(const OpFoldResult &lhs, const OpFoldResult &rhs,
                             const Location &loc, OpBuilder &b) {
  auto lhsInt = getConstantOfAttr(lhs);
  auto rhsInt = getConstantOfAttr(rhs);
  if (lhsInt && rhsInt)
    return b.getIndexAttr(std::max(lhsInt.value(), rhsInt.value()));

  auto lhsValue = dyn_cast<Value>(lhs), rhsValue = dyn_cast<Value>(rhs);
  if (lhsInt)
    lhsValue = createConstIndexValueOp(loc, b, lhsInt.value());
  else
    assert(isa<IndexType>(lhsValue.getType()));

  if (rhsInt)
    rhsValue = createConstIndexValueOp(loc, b, rhsInt.value());
  else
    assert(isa<IndexType>(rhsValue.getType()));

  return b.create<arith::MaxSIOp>(loc, lhsValue, rhsValue).getResult();
}

LogicalResult
addReduceWithIndexAttrIfNeeded(ConversionPatternRewriter &rewriter,
                               linalg::ReduceOp reduceOp) {
  // To verify whether the operation of the reduceOp is ReduceWithIndex
  // TODO: maybe a better way of judging?
  Block &body = reduceOp.getCombiner().front();
  auto yieldOp = dyn_cast<linalg::YieldOp>(body.getTerminator());

  auto yieldValue = yieldOp.getValues();
  if (yieldValue.size() == 0) {
    return failure();
  }

  auto opIter = reduceOp.getBody()->without_terminator().begin();
  auto cmpMaskOp = dyn_cast<arith::CmpFOp>(*opIter);
  const StringRef reduceRef = "reduce_mode";
  if (cmpMaskOp) {
    if (cmpMaskOp.getPredicate() == arith::CmpFPredicate::OGT) {
      reduceOp->setAttr(reduceRef, rewriter.getStringAttr("max_with_index"));
    } else if (cmpMaskOp.getPredicate() == arith::CmpFPredicate::OLT) {
      reduceOp->setAttr(reduceRef, rewriter.getStringAttr("min_with_index"));
    }
  }

  auto cmpMaskIOp = dyn_cast<arith::CmpIOp>(*opIter);
  if (cmpMaskIOp) {
    if (cmpMaskIOp.getPredicate() == arith::CmpIPredicate::sgt ||
        cmpMaskIOp.getPredicate() == arith::CmpIPredicate::ugt) {
      reduceOp->setAttr(reduceRef, rewriter.getStringAttr("max_with_index"));
    } else if (cmpMaskIOp.getPredicate() == arith::CmpIPredicate::slt ||
               cmpMaskIOp.getPredicate() == arith::CmpIPredicate::ult) {
      reduceOp->setAttr(reduceRef, rewriter.getStringAttr("min_with_index"));
    }
  }

  return success();
}

// Fold layout constant info to attr, otherwise convert to index type value
OpFoldResult getOpFoldResultOfLayoutInfo(Value value, OpBuilder &builder) {
  OpFoldResult constantFold = getAsOpFoldResult(value);
  if (llvm::isa<Attribute>(constantFold)) {
    assert(isa<IntegerAttr>(constantFold.get<Attribute>()));
    return constantFold;
  }

  if (!isa<IntegerType>(value.getType()))
    llvm_unreachable("Illegal data type when parse block data layout info");

  if (!isa<IndexType>(value.getType())) {
    if (value.getType().isInteger(/*width*/ 1))
      value = builder.create<arith::IndexCastUIOp>(
          value.getLoc(), builder.getIndexType(), value);
    else
      value = builder.create<arith::IndexCastOp>(value.getLoc(),
                                                 builder.getIndexType(), value);
  }

  return value;
}

} // namespace mlir
