//===-- TritonAscendDialect.td - dialect op definitions -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_DIALECT
#define TRITON_ASCEND_DIALECT

include "mlir/IR/OpBase.td"

def TritonAscend_Dialect : Dialect {
  let name = "ascend";
  let cppNamespace = "::mlir::triton::ascend";
  let summary = "The TritonAscend dialect in Triton.";

  let description = [{
    TritonAscend is a dialect for representing operations on Ascend NPUs.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "triton::TritonDialect",
  ];

  let extraClassDeclaration = [{}];
}

#endif // TRITON_ASCEND_DIALECT
