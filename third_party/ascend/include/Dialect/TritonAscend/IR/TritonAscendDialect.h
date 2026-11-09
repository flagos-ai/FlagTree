//===- TritonAscendDialect.h - MLIR TritonAscend dialect --------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the TritonAscend dialect in MLIR, containing Ascend
// operations.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_DIALECT_ASCEND_DIALECT_H
#define TRITON_DIALECT_ASCEND_DIALECT_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendOpsAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendOps.h.inc"

#include "mlir/Transforms/InliningUtils.h"

namespace mlir::triton::ascend {

struct TritonAscendInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;
  // All operations within the TritonAscend dialect (eg: ascend.sort,
  // ascend.flip) can be inlined.
  bool isLegalToInline(mlir::Operation *, mlir::Region *, bool,
                       mlir::IRMapping &) const final {
    return true;
  }
};

} // namespace mlir::triton::ascend

#endif // TRITON_DIALECT_ASCEND_DIALECT_H
