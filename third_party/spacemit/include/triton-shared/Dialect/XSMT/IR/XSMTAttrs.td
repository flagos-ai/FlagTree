//===- XSMTAttrs.td - XSMT dialect attrs -------------------*- tablegen -*-===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef XSMT_ATTRS
#define XSMT_ATTRS

include "mlir/IR/AttrTypeBase.td"
include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.td"
include "triton-shared/Dialect/XSMT/IR/XSMTDialect.td"

class XSMT_Attr<string name, string _mnemonic, list<Trait> traits = []>
    : AttrDef<XSMT_Dialect, name, traits> {
  let mnemonic = _mnemonic;
}

def XSMT_MemorySpaceAttr
    : XSMT_Attr<"MemorySpace", "memory_space",
                 [DeclareAttrInterfaceMethods<MemorySpaceAttrInterface>]> {
  let summary = "Memory space visibility classification";
  let description = [{
    Used on MemRefType / UnrankedMemRefType memory space slots as a logical
    visibility/storage classification. The four canonical values below match
    spine-mlir's XSMT dialect exactly so that IR emitted by spine-triton is
    accepted by spine-mlir's `memory-space-inference` and backend lowering
    passes.

    Canonical values (the only values accepted by `verify`):
      global              — module-level / externally visible buffer
                            (kernel entry memref params, memref.global)
      thread_local        — compiler-managed private scratch/stack space
                            (memref.alloc / memref.alloca default)
      cluster_shared_l2   — cluster-shared L2 scratchpad
      fragment            — register file / fragment storage; must be
                            resolved before ConvertMemRefToSpeRT

    Note: this is a logical tag, not a physical address space; spine-mlir
    maps all of these to LLVM address space 0 at FinalizeMemRefToLLVM time.

    Syntax: `#xsmt.memory_space<global>`
  }];

  let parameters = (ins StringRefParameter<"memory space name">:$value);
  let assemblyFormat = "`<` $value `>`";
  let genVerifyDecl = 1;
}

#endif // XSMT_ATTRS
