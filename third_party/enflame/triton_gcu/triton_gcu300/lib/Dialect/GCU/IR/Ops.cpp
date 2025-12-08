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
      rewriter.replaceOpWithNewOp<VectorConvertOp>(op, op->getResultTypes(),
                                                   from->getOperands());
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
  } else {
    if (dims[1] % 16 != 0 || dims[2] % 512 != 0)
      return emitOpError() << "dim1 needs align to 16 and dim2 needs align to "
                              "512 in reduce_minmax";
  }
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

} // namespace gcu
} // namespace mlir
