//===------------------- MagicKernelDialect.cpp ---------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

using namespace mlir;
using namespace mlir::mk;

LogicalResult PrintOp::verify() {
  if (getOperands().size() > 1)
    return emitOpError("expects at most one operand");
  return success();
}

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void MagicKernelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "magic-kernel/Dialect/IR/MagicKernelOps.cpp.inc"
      >();
  // TODO: Add BufferizableOpInterface to all ops that can be bufferized
  declarePromisedInterfaces<bufferization::BufferizableOpInterface, mk::DotOp,
                            mk::DotScaledOp, mk::SigmoidOp, mk::GatherOp,
                            mk::PrintOp>();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

OpFoldResult mk::BitcastOp::fold(FoldAdaptor adaptor) {
  if (getOperand().getType() == getResult().getType()) {
    return getOperand();
  }
  return {};
}

#define GET_OP_CLASSES
#include "magic-kernel/Dialect/IR/MagicKernelOps.cpp.inc"

#include "magic-kernel/Dialect/IR/MagicKernelDialect.cpp.inc"
