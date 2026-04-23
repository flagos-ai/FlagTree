/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/GCU/IR/Dialect.h"
namespace mlir {
namespace gcu {

LogicalResult AllocBarrierOp::verify() {
  if (getBarrier().getType().getAddressSpace().getValue() !=
      gcu::AddressSpace::Workgroup)
    return emitOpError() << "only supports workgroup level";
  return success();
}

LogicalResult AllocPipelineOp::verify() {
  if (getPipeline().getType().getAddressSpace().getValue() !=
      gcu::AddressSpace::Workgroup)
    return emitOpError() << "only supports workgroup level";
  PipelineType pipelineType = getPipeline().getType();
  if (pipelineType.getStageCount() <= 0 ||
      pipelineType.getProducerCount() <= 0 ||
      pipelineType.getConsumerCount() <= 0)
    return emitOpError() << "stage count, producer count and consumer count "
                            "must be greater than 0";
  return success();
}

LogicalResult DynamicSharedMemoryOp::verify() {
  if (!getOperation()->getParentWithTrait<OpTrait::SymbolTable>())
    return emitOpError() << "must be inside an op with symbol table";

  MemRefType memrefType = getResultMemref().getType();
  // Check address space
  if (auto addrspace = memrefType.getMemorySpace()) {
    if (!(dyn_cast<gpu::AddressSpaceAttr>(addrspace) &&
          dyn_cast<gpu::AddressSpaceAttr>(addrspace).getValue() ==
              gpu::AddressSpace::Workgroup) &&
        !(dyn_cast<gcu::AddressSpaceAttr>(addrspace) &&
          (dyn_cast<gcu::AddressSpaceAttr>(addrspace).getValue() ==
               gcu::AddressSpace::Workgroup ||
           dyn_cast<gcu::AddressSpaceAttr>(addrspace).getValue() ==
               gcu::AddressSpace::Local)))
      return emitOpError() << "address space must be "
                           << gpu::AddressSpaceAttr::getMnemonic() << "<"
                           << stringifyEnum(gpu::AddressSpace::Workgroup) << ">"
                           << " or " << gcu::AddressSpaceAttr::getMnemonic()
                           << "<" << stringifyEnum(gcu::AddressSpace::Workgroup)
                           << ">"
                           << " or " << gcu::AddressSpaceAttr::getMnemonic()
                           << "<" << stringifyEnum(gcu::AddressSpace::Local)
                           << ">";
  }
  if (memrefType.hasStaticShape()) {
    return emitOpError() << "result memref type must be memref<?xi8, "
                            "#gpu.address_space<workgroup>> or <?xi8, "
                            "#gcu.address_space<workgroup>> or <?xi8, "
                            "#gcu.address_space<local>>";
  }
  return success();
}

LogicalResult MemsetAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  Type value = getValue().getType();
  if (dst.getElementType() != value)
    return emitOpError() << "value type should be same as dst's element type";
  return success();
}

LogicalResult MemcpyAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  auto is1RankContious = [](MemRefType t) {
    return (t.getRank() == 1 && t.isLastDimUnitStride() &&
            !t.getLayout().getAffineMap().isConstant());
  };
  if (is1RankContious(dst) && src.getLayout().isIdentity())
    return success();
  if (dst.getLayout().isIdentity() && is1RankContious(src))
    return success();
  if (is1RankContious(dst) && is1RankContious(src))
    return success();
  if (dst.getLayout().isIdentity() && src.getLayout().isIdentity() &&
      dst.getLayout() == src.getLayout() && dst.getShape() == src.getShape() &&
      dst.getElementType() == src.getElementType())
    return success();

  return emitOpError() << "dst and src types should be 1 rank memref "
                          " or canonical form memory and with same shape";
}

LogicalResult SliceAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if ( // dst.getLayout().isIdentity() &&
       // src.getLayout().isIdentity() &&
      dst.getElementType() == src.getElementType() &&
      dst.getRank() == src.getRank() &&
      static_cast<unsigned>(dst.getRank()) == getOffsets().size() &&
      dst.getRank() <= 5)
    return success();
  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult SlicePadAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if ( // dst.getLayout().isIdentity() &&
       // src.getLayout().isIdentity() &&
      dst.getElementType() == src.getElementType() &&
      getPadValue().getType() == dst.getElementType() &&
      dst.getRank() == src.getRank() &&
      static_cast<unsigned>(dst.getRank()) == getOffsets().size() &&
      dst.getRank() <= 5)
    return success();
  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult DesliceAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if ( // dst.getLayout().isIdentity() &&
       // src.getLayout().isIdentity() &&
      dst.getElementType() == src.getElementType() &&
      dst.getRank() == src.getRank() &&
      static_cast<unsigned>(dst.getRank()) == getOffsets().size() &&
      dst.getRank() <= 5)
    return success();
  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult SliceDesliceAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if ( // dst.getLayout().isIdentity() &&
       // src.getLayout().isIdentity() &&
      dst.getElementType() == src.getElementType() &&
      dst.getRank() == src.getRank() &&
      static_cast<unsigned>(dst.getRank()) == getOffsets().size() &&
      dst.getRank() <= 5)
    return success();
  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult TransposeAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if ( // dst.getLayout().isIdentity() &&
       // src.getLayout().isIdentity() &&
      dst.getElementType() == src.getElementType() &&
      dst.getRank() == src.getRank() &&
      static_cast<unsigned>(dst.getRank()) == getLayout().size() &&
      dst.getRank() <= 5)
    return success();
  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult BroadcastAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if ( // dst.getLayout().isIdentity() &&
       // src.getLayout().isIdentity() &&
      dst.getElementType() == src.getElementType() &&
      dst.getRank() >= src.getRank() && dst.getRank() <= 5)
    return success();
  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  if (src.getRank() > dst.getRank())
    return emitOpError() << "src rank should be less or equal then dst rank";
  return emitOpError() << "dst's rank should has larger than src's, "
                          "element type and be identity memref";
}

LogicalResult SliceBroadcastAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if ( // dst.getLayout().isIdentity() &&
       // src.getLayout().isIdentity() &&
      dst.getElementType() == src.getElementType() &&
      static_cast<unsigned>(src.getRank()) == getOffsets().size() &&
      dst.getRank() >= src.getRank() && dst.getRank() <= 5)
    return success();
  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  if (src.getRank() > dst.getRank())
    return emitOpError() << "src rank should be less or equal then dst rank";
  return emitOpError() << "dst's rank should has larger than src's, "
                          "element type and be identity memref";
}

LogicalResult SliceTransposeAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if ( // dst.getLayout().isIdentity() &&
       // src.getLayout().isIdentity() &&
      dst.getElementType() == src.getElementType() &&
      static_cast<unsigned>(src.getRank()) == getOffsets().size() &&
      static_cast<unsigned>(dst.getRank()) == getLayout().size() &&
      dst.getRank() == src.getRank() && dst.getRank() <= 5)
    return success();
  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult TransposeDesliceAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if ( // dst.getLayout().isIdentity() &&
       // src.getLayout().isIdentity() &&
      dst.getElementType() == src.getElementType() &&
      static_cast<unsigned>(src.getRank()) == getLayout().size() &&
      static_cast<unsigned>(dst.getRank()) == getOffsets().size() &&
      dst.getRank() == src.getRank() && dst.getRank() <= 5)
    return success();
  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult MemsetDesliceAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  Type value = getValue().getType();

  if (src.getElementType() != value)
    return emitOpError() << "value type should be same as src's element type";

  if ( // dst.getLayout().isIdentity() &&
       // src.getLayout().isIdentity() &&
      dst.getElementType() == src.getElementType() &&
      dst.getRank() == src.getRank() &&
      static_cast<unsigned>(dst.getRank()) == getOffsets().size() &&
      dst.getRank() <= 5)
    return success();
  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult MirrortbAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if (dst.getElementType() == src.getElementType() && dst.getRank() == 2 &&
      src.getRank() == 2)
    return success();
  if (src.getRank() != 2)
    return emitOpError() << "mirror op only support 2D tensor";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult MirrorlrAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if (dst.getElementType() == src.getElementType() && dst.getRank() == 2 &&
      src.getRank() == 2)
    return success();
  if (src.getRank() != 2)
    return emitOpError() << "mirror op only support 2D tensor";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult MirrortbPadAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if (dst.getElementType() == src.getElementType() &&
      getPadValue().getType() == dst.getElementType() && dst.getRank() == 2 &&
      src.getRank() == 2)
    return success();
  if (src.getRank() != 2)
    return emitOpError() << "mirror op only support 2D tensor";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult MirrorlrPadAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if (dst.getElementType() == src.getElementType() &&
      getPadValue().getType() == dst.getElementType() && dst.getRank() == 2 &&
      src.getRank() == 2)
    return success();
  if (src.getRank() != 2)
    return emitOpError() << "mirror op only support 2D tensor";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult MirrortbDesliceAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if (dst.getElementType() == src.getElementType() && dst.getRank() == 2 &&
      src.getRank() == 2)
    return success();
  if (src.getRank() != 2)
    return emitOpError() << "mirror op only support 2D tensor";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult MirrorlrDesliceAsyncOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();
  if (dst.getElementType() == src.getElementType() && dst.getRank() == 2 &&
      src.getRank() == 2)
    return success();
  if (src.getRank() != 2)
    return emitOpError() << "mirror op only support 2D tensor";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult VectorConvertOp::verify() {
  if (getNumOperands() > getNumResults() &&
      getNumOperands() % getNumResults() != 0)
    return emitOpError() << "number of inputs should be multiply of outputs'";
  if (getNumOperands() < getNumResults() &&
      getNumResults() % getNumOperands() != 0)
    return emitOpError() << "number of outputs should be multiply of inputs'";

  uint64_t inputElems = 0;
  Type inputType;
  for (auto input : getInputs()) {
    auto t = dyn_cast<VectorType>(input.getType());
    inputElems += t.getNumElements();
    if (inputType && t != inputType)
      return emitOpError() << "all inputs' types should be same";
    inputType = t;
  }
  uint64_t outputElems = 0;
  Type outputType;
  for (auto output : getOutputs()) {
    auto t = dyn_cast<VectorType>(output.getType());
    outputElems += t.getNumElements();
    if (outputType && t != outputType)
      return emitOpError() << "all outputs' types should be same";
    outputType = t;
  }

  if (inputElems == 0)
    return emitOpError() << "inputs should not be empty";
  if (outputElems == 0)
    return emitOpError() << "outputs should not be empty";
  if (inputElems != outputElems)
    return emitOpError()
           << "inputs should have same element number with outputs";
  return success();
}

struct SimplifyRedundantVectorConvert
    : public OpRewritePattern<VectorConvertOp> {
  explicit SimplifyRedundantVectorConvert(MLIRContext *context)
      : OpRewritePattern<VectorConvertOp>(context, /*benefit=*/1) {}

  LogicalResult matchAndRewrite(VectorConvertOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    unsigned numInputs = op.getNumOperands();
    unsigned numOutputs = op.getNumResults();

    // if all inputs' types are same as outputs', just remove it
    if (numInputs == numOutputs) {
      bool isAllSame = true;
      for (unsigned i = 0; i < numInputs; ++i) {
        if (op.getOperand(i).getType() != op.getResult(i).getType()) {
          isAllSame = false;
          break;
        }
      }
      if (isAllSame) {
        for (unsigned i = 0; i < numInputs; ++i)
          rewriter.replaceAllUsesWith(op.getResult(i), op.getOperand(i));
        rewriter.eraseOp(op);
        return success();
      }
    }

    // if inputs are from type conversion ops, just remove it
    if (numInputs == numOutputs) {
      auto isCvtOp = [](Operation *op) {
        return isa<arith::UIToFPOp, arith::SIToFPOp, arith::ExtSIOp,
                   arith::ExtUIOp, arith::ExtFOp, arith::TruncIOp,
                   arith::TruncFOp, arith::IndexCastOp>(op);
      };
      auto isValidCvt = [](Operation *op, Type from, Type to) {
        auto fromTy = dyn_cast<VectorType>(from);
        if (!fromTy)
          return false;
        auto toTy = dyn_cast<VectorType>(to);
        if (!toTy)
          return false;
        if (isa<arith::UIToFPOp, arith::SIToFPOp, arith::IndexCastOp>(op))
          return true;
        if (isa<arith::ExtSIOp, arith::ExtUIOp, arith::ExtFOp>(op)) {
          return fromTy.getElementTypeBitWidth() <=
                 toTy.getElementTypeBitWidth();
        }
        return fromTy.getElementTypeBitWidth() >= toTy.getElementTypeBitWidth();
      };

      bool isAllSame = true;
      SmallVector<Operation *, 4> cvtOps;
      for (unsigned i = 0; i < numInputs; ++i) {
        if (!op.getOperand(i).getDefiningOp() ||
            !isCvtOp(op.getOperand(i).getDefiningOp())) {
          isAllSame = false;
          break;
        }
        auto cvtOp = op.getOperand(i).getDefiningOp();
        if (!isValidCvt(cvtOp, cvtOp->getOperand(0).getType(),
                        op.getResult(i).getType())) {
          isAllSame = false;
          break;
        }
        cvtOps.push_back(cvtOp);
        if (cvtOps.front()->getName() != cvtOp->getName()) {
          isAllSame = false;
          break;
        }
      }
      if (isAllSame) {
        for (unsigned i = 0; i < numInputs; ++i) {
          auto newCvtOp = rewriter.clone(*cvtOps[i]);
          newCvtOp->getResult(0).setType(op.getResult(i).getType());
          rewriter.replaceAllUsesWith(op.getResult(i), newCvtOp->getResult(0));
        }
        rewriter.eraseOp(op);
        return success();
      }
    }

    // check if there are two converts in chain
    bool isOperandFromSameVectorConvert = true;
    Operation *from = nullptr;
    for (unsigned i = 0; i < numInputs; ++i) {
      auto v = op.getOperand(i);
      if (!v.getDefiningOp()) {
        isOperandFromSameVectorConvert = false;
        break;
      }
      if (from && from != v.getDefiningOp()) {
        isOperandFromSameVectorConvert = false;
        break;
      }
      from = v.getDefiningOp();
      if (!isa<VectorConvertOp>(from)) {
        isOperandFromSameVectorConvert = false;
        break;
      }
    }
    if (!from)
      isOperandFromSameVectorConvert = false;
    if (from && from->getNumResults() != numInputs)
      isOperandFromSameVectorConvert = false;
    for (unsigned i = 0; i < numInputs && isOperandFromSameVectorConvert; ++i) {
      if (i >= from->getNumResults() || op.getOperand(i) != from->getResult(i))
        isOperandFromSameVectorConvert = false;
    }
    if (isOperandFromSameVectorConvert) {
      // The backend only permits i1<->i8 conversions for boolean vectors.
      // When both sides of the merged convert carry i1 elements, only allow
      // folding if the result is an identity (same count and matching types),
      // which will then be cleaned up by identity removal.  Block non-identity
      // folds that would produce illegal patterns such as:
      //   gcu.vector_convert(v256i1, v256i1) -> v512i1   (INVALID)
      //   gcu.vector_convert(v512i1) -> (v256i1, v256i1) (INVALID)
      auto hasI1Elem = [](TypeRange types) {
        return llvm::any_of(types, [](Type type) {
          return getElementTypeOrSelf(type).isInteger(1);
        });
      };
      if (hasI1Elem(from->getOperandTypes()) &&
          hasI1Elem(op->getResultTypes())) {
        bool isIdentity = from->getNumOperands() == op->getNumResults();
        for (unsigned i = 0; i < from->getNumOperands() && isIdentity; ++i)
          isIdentity =
              from->getOperand(i).getType() == op->getResult(i).getType();
        if (!isIdentity)
          return failure();
      }

      // rewriter.replaceOpWithNewOp<VectorConvertOp>(op, op->getResultTypes(),
      //                                              from->getOperands());
      auto newOp = rewriter.create<VectorConvertOp>(
          op.getLoc(), op->getResultTypes(), from->getOperands());
      rewriter.replaceOp(op, newOp);
      return success();
    }

    // split convert if possible
    unsigned times = numOutputs > numInputs ? numInputs : numOutputs;

    if (times <= 1)
      return failure();

    unsigned inputStep = numInputs / times;
    unsigned outputStep = numOutputs / times;
    for (unsigned i = 0; i < times; ++i) {
      SmallVector<Value, 4> inputs;
      for (unsigned j = i * inputStep; j < i * inputStep + inputStep; ++j)
        inputs.push_back(op.getOperand(j));
      SmallVector<Value, 4> outputs;
      SmallVector<Type, 4> outputTypes;
      for (unsigned j = i * outputStep; j < i * outputStep + outputStep; ++j) {
        outputs.push_back(op.getResult(j));
        outputTypes.push_back(outputs.back().getType());
      }
      auto convert = rewriter.create<VectorConvertOp>(loc, outputTypes, inputs);
      for (unsigned j = 0; j < outputStep; ++j) {
        rewriter.replaceAllUsesWith(outputs[j], convert.getResult(j));
      }
    }
    rewriter.eraseOp(op);
    return success();
  }
};

void VectorConvertOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                  MLIRContext *context) {
  results.add<SimplifyRedundantVectorConvert>(context);
}

LogicalResult PtrToMemRefOp::verify() {
  auto memrefType = getResultMemref().getType();
  if (memrefType.getMemorySpace())
    return emitOpError() << "result memref type should not has memory space";
  if (memrefType.hasStaticShape()) {
    return emitOpError() << "result memref type must be memref<?x...>";
  }
  // if (getPtr().getType().getElementType() != memrefType.getElementType())
  //   return emitOpError() << "pointer element type must be same as result's";
  return success();
}

LogicalResult MemRefToPtrOp::verify() {
  auto memrefType = getMemref().getType();
  if (!memrefType.getLayout().isIdentity()) {
    return emitOpError() << "memref type must have identity layout";
  }
  // if (memrefType.getMemorySpace())
  //   return emitOpError() << "memref type should not has memory space";
  // if (memrefType.hasStaticShape()) {
  //   return emitOpError() << "memref type must be memref<?x...>";
  // }
  // if (getPtr().getType().getElementType() != memrefType.getElementType())
  //   return emitOpError() << "pointer element type must be same as input's";
  return success();
}

LogicalResult MatMulOp::verify() {
  MemRefType out = getOut().getType();
  MemRefType lhs = getLhs().getType();
  MemRefType rhs = getRhs().getType();

  if (lhs.getElementType() != rhs.getElementType())
    return emitOpError()
           << "element type of operands lhs and rhs must be same type";
  if (lhs.getRank() != rhs.getRank() || out.getRank() != lhs.getRank())
    return emitOpError() << "out, lhs and rhs types should have same rank";
  if (out.getRank() != 2 && out.getRank() != 3)
    return emitOpError() << "rank must be 2D or 3D";
  else if (out.getRank() == 3 &&
           getLhs().getType().getShape()[0] != getRhs().getType().getShape()[0])
    return emitOpError() << "lhs[dim0=b, dim1=m, dim2=k] and rhs[dim0=b, "
                            "dim1=k, dim2=n] must have the same dim0";
  // add bias check
  if (getBias()) {
    if (getBias().getType().getShape()[0] != out.getShape()[0] ||
        getBias().getType().getShape()[1] != out.getShape()[1]) {
      return emitOpError() << "out and bias should have same shape!!!!";
    }
  }
  return success();
}

LogicalResult ReduceOp::verify() {
  auto dims = getIn().getType().getShape();
  ReduceOperation op = getOp();
  if (op == ReduceOperation::SUM) {
    if (dims[1] % 128 != 0 || dims[2] % 128 != 0)
      return emitOpError()
             << "both dim1 and dim2 need align to 128 in reduce_sum";
  } else if (op == ReduceOperation::MEAN) {
    auto axis = getAxis();
    if (axis == 2) {
      if (dims[1] % 128 != 0 || dims[2] % 128 != 0)
        return emitOpError()
               << "both dim1 and dim2 need align to 128 in reduce_mean";
      return success();
    } else if (axis == 1) {
      if (dims[1] % 16 != 0 || dims[2] % 128 != 0)
        return emitOpError() << "dim1 need align to 16 and dim2 need align"
                             << " to 128 in reduce_mean";
      return success();
    } else {
      return emitOpError() << "just support axis 1 or 2";
    }
  } else {
    if (dims[1] % 16 != 0 || dims[2] % 512 != 0)
      return emitOpError() << "dim1 needs align to 16 and dim2 needs align to "
                              "512 in reduce_minmax";
  }
  return success();
}

LogicalResult ReduceArgOp::verify() {
  auto dims = getIn().getType().getShape();
  if (dims[1] % 16 != 0 || dims[2] % 128 != 0)
    return emitOpError() << "dim1 needs align to 16 and dim2 needs align to "
                            "512 in reduce_minmax";
  return success();
}

LogicalResult AtomicRMWOp::verify() {
  auto rmw_op = getAtomicRmwOp();
  auto type = getVal().getType();
  auto element_type = getVal().getType().isIntOrFloat()
                          ? type
                          : dyn_cast<mlir::TensorType>(type).getElementType();
  auto memory_sync_scope = getScope();
  auto bitwidth = element_type.getIntOrFloatBitWidth();

  // check supported data type
  if (rmw_op == gcu::RMWOp::ADD) {
    if (8 == bitwidth)
      return emitOpError()
             << "only supports i16/u16/i32/u32/i64/u64/fp32/fp16/bf16";
  } else if (rmw_op == gcu::RMWOp::MAX || rmw_op == gcu::RMWOp::UMAX) {
    if (8 == bitwidth || 16 == bitwidth)
      return emitOpError() << "only supports i32/u32/i64/u64/fp32/fp16/bf16";
  } else if (rmw_op == gcu::RMWOp::MIN || rmw_op == gcu::RMWOp::UMIN) {
    if (8 == bitwidth || 16 == bitwidth || element_type.isF32())
      return emitOpError() << "only supports i32/u32/i64/u64";
  } else if (rmw_op == gcu::RMWOp::AND) {
    if (8 == bitwidth || 16 == bitwidth || element_type.isF32())
      return emitOpError() << "only supports i32/u32/i64/u64";
  } else if (rmw_op == gcu::RMWOp::OR) {
    if (8 == bitwidth || 16 == bitwidth || element_type.isF32())
      return emitOpError() << "only supports i32/u32/i64/u64";
  } else if (rmw_op == gcu::RMWOp::XOR) {
    if (8 == bitwidth || 16 == bitwidth || element_type.isF32())
      return emitOpError() << "only supports i32/u32/i64/u64";
  } else if (rmw_op == gcu::RMWOp::XCHG) {
    if (8 == bitwidth || element_type.isBF16())
      return emitOpError() << "only supports i16/u16/i32/u32/i64/u64/fp32/fp16";
  }

  // check supported memory sync scope
  if (!(memory_sync_scope == gcu::MemSyncScope::GCU))
    return emitOpError() << "only supports atomic memory sync scope is gcu";

  return success();
}

LogicalResult AtomicCASOp::verify() {
  auto type = getVal().getType();
  auto element_type = getVal().getType().isIntOrFloat()
                          ? type
                          : dyn_cast<mlir::TensorType>(type).getElementType();
  auto memory_sync_scope = getScope();
  auto bitwidth = element_type.getIntOrFloatBitWidth();

  // check supported data type
  if (8 == bitwidth || 16 == bitwidth || element_type.isF32())
    return emitOpError() << "only supports i32/u32/i64/u64";

  // check supported memory sync scope
  if (!(memory_sync_scope == gcu::MemSyncScope::GCU))
    return emitOpError() << "only supports atomic memory sync scope is gcu";

  return success();
}

RegionRange WarpSpecializeOp::getPartitionRegions() {
  return cast<WarpSpecializePartitionsOp>(
             getPartitionOpHolder().front().front())
      .getPartitionRegions();
}

void WarpSpecializeOp::getSuccessorRegions(
    RegionBranchPoint src, SmallVectorImpl<RegionSuccessor> &successors) {
  // The parent branches transparently into the default region.
  if (src.isParent()) {
    successors.emplace_back(&getDefaultRegion());
    return;
  }
  // And the default region branches transparently back to the parent.
  assert(src.getRegionOrNull() == &getDefaultRegion());
  successors.push_back(RegionSuccessor(getResults()));
}

LogicalResult WarpSpecializeOp::verify() {
  // The default region is not isolated from above but the partition regions
  // have to be. MLIR does not support this, so we hide an op inside another
  // region that contains the isolated regions. Check that it is there.
  if (!isa<WarpSpecializePartitionsOp>(
          getPartitionOpHolder().front().front())) {
    return emitOpError(
        "expected to find only a `gcu.warp_specialize.partitions` op inside "
        "its second region");
  }

  // Verify the partitions.
  if (getPartitionRegions().size() != getPartitionNumWarps().size()) {
    return emitOpError("has ") << getPartitionRegions().size()
                               << " partitions but `partitionNumWarps` has "
                               << getPartitionNumWarps().size() << " elements";
  }
  for (auto [i, numWarps] : llvm::enumerate(getPartitionNumWarps())) {
    if (llvm::isPowerOf2_32(numWarps))
      continue;
    return emitOpError("partition #")
           << i << " number of warps (" << numWarps << ") must be a power of 2";
  }
  if (std::optional<ArrayRef<int32_t>> startIds = getWarpGroupStartIds()) {
    if (startIds->size() != getPartitionNumWarps().size()) {
      return emitOpError("has ")
             << startIds->size() << " warp group start IDs but expected "
             << getPartitionNumWarps().size();
    }
  }

  for (auto [i, region] : llvm::enumerate(getPartitionRegions())) {
    if (region->getNumArguments() != getNumOperands()) {
      return emitOpError("partition region #")
             << i << " has " << region->getNumArguments()
             << " arguments but expected " << getNumOperands();
    }
    for (auto [argIdx, argType, capType] : llvm::enumerate(
             region->getArgumentTypes(), getExplicitCaptures().getTypes())) {
      if (argType == capType)
        continue;
      return emitOpError("partition region #")
             << i << " argument #" << argIdx << " has type " << argType
             << " but corresponding capture has type " << capType;
    }
  }

  // This op cannot be nested inside itself.
  if ((*this)->getParentOfType<WarpSpecializeOp>()) {
    return emitOpError(
        "cannot be nested inside another `gcu.warp_specialize` op");
  }

  // std::optional<int> numWarps = maybeLookupNumWarps(*this);
  // if (numWarps && *numWarps % 4 != 0) {
  //   return mlir::emitError(getLoc()) << "warp-specialized kernels requires "
  //                                       "num_warps to be a multiple of 4";
  // }

  return success();
}

LogicalResult WarpSpecializeOp::canonicalize(WarpSpecializeOp op,
                                             PatternRewriter &b) {
  // Propagate unused results and captures by removing them from the op.
  llvm::BitVector unusedArgs(op.getNumOperands());
  llvm::BitVector unusedResults(op.getNumResults());
  for (auto [i, result] : llvm::enumerate(op.getResults())) {
    if (result.use_empty())
      unusedResults.set(i);
  }
  // Remove duplicate captures.
  DenseMap<Value, unsigned> uniqueCaptures;
  for (auto [i, capture] : llvm::enumerate(op.getExplicitCaptures())) {
    auto noUseInRegion = [i = i](Region *region) {
      return region->getArgument(i).use_empty();
    };
    if (llvm::all_of(op.getPartitionRegions(), noUseInRegion)) {
      unusedArgs.set(i);
      continue;
    }

    auto [it, inserted] = uniqueCaptures.try_emplace(capture, i);
    if (!inserted) {
      unsigned duplicateIdx = it->second;
      b.modifyOpInPlace(op, [&, i = i] {
        for (Region *region : op.getPartitionRegions()) {
          b.replaceAllUsesWith(region->getArgument(i),
                               region->getArgument(duplicateIdx));
        }
      });
      unusedArgs.set(i);
    }
  }
  if (unusedArgs.none() && unusedResults.none())
    return failure();

  if (unusedArgs.any()) {
    b.modifyOpInPlace(op, [&] {
      for (Region *region : op.getPartitionRegions())
        region->front().eraseArguments(unusedArgs);
      op->eraseOperands(unusedArgs);
    });
  }

  if (unusedResults.any()) {
    for (Block &block : op.getDefaultRegion()) {
      if (auto yield = dyn_cast<WarpYieldOp>(block.getTerminator())) {
        b.modifyOpInPlace(yield, [&] { yield->eraseOperands(unusedResults); });
      }
    }

    SmallVector<Type> newTypes;
    for (auto [i, type] : llvm::enumerate(op.getResultTypes())) {
      if (!unusedResults.test(i))
        newTypes.push_back(type);
    }
    OperationState state(op.getLoc(), op->getName(), op.getOperands(), newTypes,
                         op->getAttrs());
    state.addRegion()->takeBody(op.getDefaultRegion());
    state.addRegion()->takeBody(op.getPartitionOpHolder());
    auto newOp = cast<WarpSpecializeOp>(b.create(state));
    unsigned newResultIdx = 0;
    for (auto [i, result] : llvm::enumerate(op.getResults())) {
      if (!unusedResults.test(i))
        result.replaceAllUsesWith(newOp.getResult(newResultIdx++));
    }
    assert(newResultIdx == newOp.getNumResults());
    b.eraseOp(op);
  }

  return success();
}

void WarpSpecializeOp::build(OpBuilder &builder, OperationState &state,
                             TypeRange resultTypes,
                             ArrayRef<int32_t> partitionNumWarps,
                             unsigned partitionNumRegions) {
  build(builder, state, resultTypes, /*explicitCaptures=*/ValueRange(),
        partitionNumWarps, {}, {}, {});
  OpBuilder::InsertionGuard guard(builder);
  builder.create<WarpSpecializePartitionsOp>(state.location,
                                             partitionNumRegions);
}

void WarpSpecializeOp::build(OpBuilder &builder, OperationState &state,
                             TypeRange resultTypes, ValueRange explicitCaptures,
                             ArrayRef<int32_t> partitionNumWarps) {
  build(builder, state, resultTypes, explicitCaptures, partitionNumWarps, {},
        {}, {});
}

ParseResult WarpSpecializeOp::parse(OpAsmParser &p, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  SMLoc operandLoc = p.getCurrentLocation();
  if (p.parseOperandList(operands, AsmParser::Delimiter::Paren) ||
      p.parseOptionalAttrDictWithKeyword(result.attributes) ||
      p.parseKeyword("default") || p.parseRegion(*result.addRegion()))
    return failure();

  OperationState partitionOpState(
      p.getEncodedSourceLoc(p.getCurrentLocation()),
      WarpSpecializePartitionsOp::getOperationName());

  SmallVector<int32_t> partitionNumWarps;
  SmallVector<OpAsmParser::Argument> partitionArgs;
  while (succeeded(p.parseOptionalKeyword(
      ("partition" + Twine(partitionNumWarps.size()).str())))) {
    partitionArgs.clear();
    if (p.parseArgumentList(partitionArgs, AsmParser::Delimiter::Paren,
                            /*allowType=*/true) ||
        p.parseKeyword("num_warps") || p.parseLParen() ||
        p.parseInteger(partitionNumWarps.emplace_back()) || p.parseRParen() ||
        p.parseRegion(*partitionOpState.addRegion(), partitionArgs))
      return failure();
  }

  FunctionType types;
  if (p.parseColon() || p.parseType(types) ||
      p.resolveOperands(operands, types.getInputs(), operandLoc,
                        result.operands))
    return failure();

  result.addTypes(types.getResults());
  result.addAttribute(getPartitionNumWarpsAttrName(result.name),
                      p.getBuilder().getDenseI32ArrayAttr(partitionNumWarps));

  Block &holder = result.addRegion()->emplaceBlock();
  OpBuilder b(p.getContext());
  b.setInsertionPointToStart(&holder);
  b.create(partitionOpState);
  return success();
}

void WarpSpecializeOp::print(OpAsmPrinter &p) {
  p << '(';
  p.printOperands(getOperands());
  p << ')';
  p.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                     {getPartitionNumWarpsAttrName()});

  p.printNewline();
  p << "default ";
  p.printRegion(getDefaultRegion(), /*printEntryBlockArgs=*/false);

  for (auto [i, region, numWarps] :
       llvm::enumerate(getPartitionRegions(), getPartitionNumWarps())) {
    p.printNewline();
    p << "partition" << i << '(';
    llvm::interleaveComma(region->getArguments(), p, [&](BlockArgument arg) {
      p.printRegionArgument(arg);
    });
    p << ") num_warps(" << numWarps << ") ";
    p.printRegion(*region, /*printEntryBlockArgs=*/false);
  }
  p << " : ";
  p.printFunctionalType(*this);
}

LogicalResult WarpYieldOp::verify() {
  if (getNumOperands() != getParentOp().getNumResults()) {
    return emitOpError("has ")
           << getNumOperands() << " operands but parent op expected "
           << getParentOp().getNumResults();
  }
  for (auto [i, result, type] :
       llvm::enumerate(getParentOp().getResultTypes(), getOperandTypes())) {
    if (result != type) {
      return emitOpError("operand #") << i << " has type " << type
                                      << " but parent op expected " << result;
    }
  }
  return success();
}

} // namespace gcu
} // namespace mlir
