//===- TritonAscendDialect.cpp - TritonAscend Dialect registration --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::triton::ascend;

void TritonAscendDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendOpsAttrDefs.cpp.inc"
      >();
}

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendOpsAttrDefs.cpp.inc"
#define GET_OP_CLASSES
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendOps.cpp.inc"
