//===-- TritonAscendAttrDefs.td - dialect attributes def.  ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_ASCEND_ATTRDEFS
#define TRITON_ASCEND_ATTRDEFS

include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class TritonAscend_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<TritonAscend_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
  let cppNamespace = "::mlir::triton::ascend";
}



#endif // TRITON_ASCEND_ATTRDEFS
