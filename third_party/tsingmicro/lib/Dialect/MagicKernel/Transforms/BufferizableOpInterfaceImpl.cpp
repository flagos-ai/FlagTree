//===- BufferizableOpInterfaceImpl.cpp ----------------------------------- ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file implements mk dialect DestinationStyleOp BufferizableOpInterface.
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Transforms/BufferizableOpInterfaceImpl.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

using namespace mlir;
using namespace mlir::bufferization;

/// Generic conversion for any DestinationStyleOpInterface on tensors.
static LogicalResult
bufferizeDestinationStyleOpInterface(RewriterBase &rewriter,
                                     DestinationStyleOpInterface op,
                                     const BufferizationOptions &options) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasPureBufferSemantics())
    return success();

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasPureTensorSemantics())
    return op->emitError() << "op does not have pure tensor semantics";

  // New input operands for the cloned op.
  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(op.getNumDpsInputs());
  for (OpOperand *opOperand : op.getDpsInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    FailureOr<Value> buffer = getBuffer(rewriter, opOperand->get(), options);
    if (failed(buffer))
      return failure();
    newInputBuffers.push_back(*buffer);
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : op->getOpResults()) {
    OpOperand *opOperand = op.getDpsInitOperand(opResult.getResultNumber());
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, opOperand->get(), options);
    if (failed(resultBuffer))
      return failure();
    newOutputBuffers.push_back(*resultBuffer);
  }

  // Merge input/output operands.
  SmallVector<Value> newOperands = newInputBuffers;
  newOperands.append(newOutputBuffers.begin(), newOutputBuffers.end());

  // Set insertion point now that potential alloc/dealloc are introduced.
  rewriter.setInsertionPoint(op);
  // Clone the op, but use the new operands. Move the existing block into the
  // new op. Since the new op does not have any tensor results, it does not
  // return anything.
  OperationState state(op->getLoc(), op->getName(), newOperands, TypeRange{},
                       op->getAttrs());

  Operation *newOp = Operation::create(state);

  // We don't want the rewriter tracks an incomplete operation, so insert new
  // operation after op was fully constructed.
  rewriter.insert(newOp);

  // Replace the results of the old op with the new output buffers.
  replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

  return success();
}

/// Bufferization of mk ops. Replace with a new mk op that operates entirely on
/// memrefs.
template <typename OpTy>
struct MKOpInterface
    : public DstBufferizableOpInterfaceExternalModel<MKOpInterface<OpTy>,
                                                     OpTy> {

  bool bufferizesToElementwiseAccess(Operation *op, const AnalysisState &state,
                                     ArrayRef<OpOperand *> opOperands) const {
    return op->hasTrait<OpTrait::Elementwise>();
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
  }
};

struct AtomicRMWOpInterface
    : public DstBufferizableOpInterfaceExternalModel<AtomicRMWOpInterface,
                                                     mk::AtomicRMWOp> {
  // TODO: Check for memory effect
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {

    auto atomicRMWOp = cast<mk::AtomicRMWOp>(op);
    if (!isa<MemRefType>(atomicRMWOp.getPtr().getType())) {
      return failure();
    }

    FailureOr<Value> valBuffer =
        getBuffer(rewriter, atomicRMWOp.getVal(), options);
    if (failed(valBuffer))
      return failure();
    FailureOr<Value> outputBuffer =
        getBuffer(rewriter, atomicRMWOp.getDst(), options);
    if (failed(outputBuffer))
      return failure();
    rewriter.create<mk::AtomicRMWOp>(
        atomicRMWOp.getLoc(),
        /*result=*/TypeRange(), atomicRMWOp.getPtr(), *valBuffer, *outputBuffer,
        atomicRMWOp.getAtomicRmwOpAttr(), atomicRMWOp.getSemAttr(),
        atomicRMWOp.getScopeAttr());
    replaceOpWithBufferizedValues(rewriter, op, *outputBuffer);
    return success();
  }
};

struct AtomicCASOpInterface
    : public DstBufferizableOpInterfaceExternalModel<AtomicCASOpInterface,
                                                     mk::AtomicCASOp> {
  // TODO: Check for memory effect
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {

    auto atomicCASOp = cast<mk::AtomicCASOp>(op);
    if (!isa<MemRefType>(atomicCASOp.getPtr().getType())) {
      return failure();
    }
    FailureOr<Value> cmpBuffer =
        getBuffer(rewriter, atomicCASOp.getCmp(), options);
    if (failed(cmpBuffer))
      return failure();

    FailureOr<Value> valBuffer =
        getBuffer(rewriter, atomicCASOp.getVal(), options);
    if (failed(valBuffer))
      return failure();

    FailureOr<Value> outputBuffer =
        getBuffer(rewriter, atomicCASOp.getDst(), options);
    if (failed(outputBuffer))
      return failure();
    rewriter.create<mk::AtomicCASOp>(
        atomicCASOp.getLoc(),
        /*result=*/TypeRange(), atomicCASOp.getPtr(), *cmpBuffer, *valBuffer,
        *outputBuffer, atomicCASOp.getSemAttr(), atomicCASOp.getScopeAttr());
    replaceOpWithBufferizedValues(rewriter, op, *outputBuffer);
    return success();
  }
};

struct BitCastOpInterface
    : public BufferizableOpInterface::ExternalModel<BitCastOpInterface,
                                                    mk::BitcastOp> {

  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getResult(0), BufferRelation::Equivalent}};
  }

  // TODO: Check for memory effect
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {

    auto bitcastOp = cast<mk::BitcastOp>(op);

    auto inputType = bitcastOp.getSrc().getType();
    auto resType = bitcastOp.getType();
    assert(isa<RankedTensorType>(inputType) && isa<RankedTensorType>(resType) &&
           "expected ranked tensor type");

    FailureOr<Value> srcBuffer =
        getBuffer(rewriter, bitcastOp.getSrc(), options);
    if (failed(srcBuffer))
      return failure();

    // Result type should have same layout and address space as the source type.
    auto sourceType = srcBuffer->getType();
    assert(isa<MemRefType>(sourceType) &&
           "expected memref type for bitcast source");
    auto rankedMemRefType = cast<MemRefType>(sourceType);

    auto resultTensorType = cast<RankedTensorType>(resType);
    MemRefType resultType = MemRefType::get(
        resultTensorType.getShape(), resultTensorType.getElementType(),
        rankedMemRefType.getLayout(), rankedMemRefType.getMemorySpace());

    replaceOpWithNewBufferizedOp<mk::BitcastOp>(rewriter, op, resultType,
                                                *srcBuffer);
    return success();
  }
};

/// Helper structure that iterates over all mkOps in `OpTys` and registers
/// the `BufferizableOpInterface` with each of them.
template <typename... Ops> struct MKOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<MKOpInterface<Ops>>(*ctx), ...);
  }
};

void mlir::mk::registerBufferizableOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::mk::MagicKernelDialect *dialect) {
        // TODO: Register all mk ops.
        MKOpInterfaceHelper<mk::DotOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::DotScaledOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::SigmoidOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::GeluOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::GatherOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::PrintOp>::registerOpInterface(ctx);
        mk::AtomicRMWOp::attachInterface<AtomicRMWOpInterface>(*ctx);
        mk::AtomicCASOp::attachInterface<AtomicCASOpInterface>(*ctx);
        MKOpInterfaceHelper<mk::ArgMaxOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::ArgMinOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::Bit2FpOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::MaskMoveOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::UnEqualVV>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::EqualVV>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::EqualVS>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::LessThenVS>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::BoolEqualVS>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::ReduceMaxOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::ReduceMinOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::ReduceSumOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::DequantOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::AddVS>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::SubVS>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::MulVS>::registerOpInterface(ctx);
        mk::BitcastOp::attachInterface<BitCastOpInterface>(*ctx);
      });
}
