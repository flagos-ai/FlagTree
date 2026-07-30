// ===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: (c) 2024 SpacemiT
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"
#include <llvm/Support/Debug.h>

#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.cpp.inc"
#define GET_OP_CLASSES
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsync.cpp.inc"

using namespace mlir;
using namespace mlir::xsmt_async;
// Register the operations that are specific to the XSMTAsync dialect.
void XSMTAsyncDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsync.cpp.inc"
      >();
}

void MBarrierAllocOp::build(OpBuilder &builder, OperationState &result,
                            Value parity, Value arr_count, Value tx_count,
                            Value ex_count) {
  result.addOperands(parity);
  result.addOperands(arr_count);
  result.addOperands(tx_count);
  result.addOperands(ex_count);
}
