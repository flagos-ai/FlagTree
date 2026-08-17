//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.cpp.inc"
#include "triton/Dialect/Triton/IR/Interfaces.h"

using namespace mlir;
using namespace mlir::xsmt;

void mlir::xsmt::XSMTDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/XSMT/IR/XSMTOps.cpp.inc"
      >();
  addInterfaces<mlir::triton::TritonInlinerInterface>();
}
