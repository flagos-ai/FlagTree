//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES
#define TRITON_TO_LINALG_EXPERIMENTAL_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToLinalgExperimental : Pass<"triton-to-linalg-experimental", "mlir::ModuleOp"> {
  let summary = "Convert Triton to Linalg dialect";
  let constructor = "triton::createTritonToLinalgExperimentalPass()";
  let options = [
      Option<"enableMakeGatherScatterTensorPtr", "enable-make-gather-scatter", "bool", /*default*/"true",
             "Enable make_gather_scatter_tptr support">,
      Option<"enableCollapseShape", "enable-collapse-shape", "bool", /*default*/"false",
             "Enable collapse shape pass">,
  ];
}

def ReconcilePtrCasts : Pass<"reconcile-ptr-casts", "mlir::ModuleOp"> {
  let summary = "Convert unrealized_cast op between tt.ptr or ptr.ptr to memref to to_memref or from_memref";
  let constructor = "triton::createReconcilePtrCastsPass()";
}

def TritonToPtr : Pass<"triton-to-ptr", "mlir::ModuleOp"> {
  let summary = "Convert Triton ops on pointers to the Ptr dialect";
  let constructor = "triton::createTritonToPtrPass()";
}

def ReconcileLlvmPtrCasts : Pass<"reconcile-llvmptr-casts", "mlir::ModuleOp"> {
  let summary = "Convert unrealized_cast op llvm.ptr";
  let constructor = "triton::createReconcileLlvmPtrCastsPass()";
}

def ScfbufferStandardized : Pass<"scf-buffer-standardized", "mlir::ModuleOp"> {
  let summary = "To standardized the scf buffer";
  let description = [{
    Preprocess `scf.for` loops to support loops where a yielded tensor is not equivalent to the respective iter_arg.
  }];
  let constructor = "triton::createScfbufferStandardizedPass()";
}

def ConvertScanOp : Pass<"convert-triton-scan", "mlir::ModuleOp"> {
    let summary = "Convert Triton ScanOp.";
    let description = [{

    }];
    let constructor = "triton::createConvertScanOpPass()";
}

#endif
