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

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    return bufferizeDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options);
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
        MKOpInterfaceHelper<mk::GatherOp>::registerOpInterface(ctx);
        MKOpInterfaceHelper<mk::PrintOp>::registerOpInterface(ctx);
      });
}
