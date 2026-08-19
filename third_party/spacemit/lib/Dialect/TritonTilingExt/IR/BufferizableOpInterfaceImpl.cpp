//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/IR/DstBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

using namespace mlir;
using namespace linalg;
using namespace mlir::bufferization;

//
// This file implements the bufferizable interface for TritonTilingExtOps.
// The interface is required for bufferization (i.e: converting from tensors to
// memrefs).
// Since the bufferization semantics of TritonTilingExtOps are identical to
// linalg ops, the implementation was borrowed almost verbatim from
// mlir/lib/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.cpp
// with the exception that the code to handle linalg's region has been removed.
// (the original implementation is in an anonymous namespace, so we cannot
// reuse)
//
namespace {

/// Generic conversion for any DestinationStyleOpInterface on tensors.
static LogicalResult bufferizeTritonTilingExtDestinationStyleOpInterface(
    RewriterBase &rewriter, DestinationStyleOpInterface op,
    const BufferizationOptions &options, BufferizationState &state) {
  // Take a guard before anything else.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  // Nothing to do. This op is already bufferized.
  if (op.hasPureBufferSemantics())
    return success();

  // Ensure op has only tensors. Allow mixed tensor-buffer mode on a per-need
  // basis.
  if (!op.hasPureTensorSemantics())
    return op->emitError() << "op does not have tensor semantics";

  // New input operands for the cloned op.
  SmallVector<Value> newInputBuffers;
  newInputBuffers.reserve(op.getNumDpsInputs());
  for (OpOperand *opOperand : op.getDpsInputOperands()) {
    if (op.isScalar(opOperand)) {
      newInputBuffers.push_back(opOperand->get());
      continue;
    }
    FailureOr<Value> buffer =
        getBuffer(rewriter, opOperand->get(), options, state);
    if (failed(buffer))
      return failure();
    newInputBuffers.push_back(*buffer);
  }

  // New output operands for the cloned op.
  SmallVector<Value> newOutputBuffers;
  for (OpResult opResult : op->getOpResults()) {
    OpOperand *opOperand = op.getDpsInitOperand(opResult.getResultNumber());
    FailureOr<Value> resultBuffer =
        getBuffer(rewriter, opOperand->get(), options, state);
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
  clone(rewriter, op, /*resultTypes=*/TypeRange{}, newOperands);

  // Replace the results of the old op with the new output buffers.
  replaceOpWithBufferizedValues(rewriter, op, newOutputBuffers);

  return success();
}

template <typename OpTy>
struct TritonTilingExtOpInterface
    : public DstBufferizableOpInterfaceExternalModel<
          TritonTilingExtOpInterface<OpTy>, OpTy> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    // Operand is read if it is used in the computation.
    return cast<DestinationStyleOpInterface>(op).isDpsInput(&opOperand);
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    // Operand is written to if it is not an input/init.
    return cast<DestinationStyleOpInterface>(op).isDpsInit(&opOperand);
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options,
                          BufferizationState &state) const {
    return bufferizeTritonTilingExtDestinationStyleOpInterface(
        rewriter, cast<DestinationStyleOpInterface>(op), options, state);
  }
};

template <typename... Ops> struct TritonTilingExtOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<TritonTilingExtOpInterface<Ops>>(*ctx), ...);
  }
};
} // namespace

void mlir::ttx::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  // clang-format off
  registry.addExtension(+[](MLIRContext *ctx, ttx::TritonTilingExtDialect *dialect) {
    TritonTilingExtOpInterfaceHelper<
      ttx::CumSumOp
    >::registerOpInterface(ctx);
  });
  // clang-format on
}
