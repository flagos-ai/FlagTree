//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//

#ifndef MEMREF_TO_MK_CONVERSION_PASSES
#define MEMREF_TO_MK_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def Tx81MemrefToLLVM : Pass<"tx81-memref-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert memref and bufferization operations into custom llvm function call.";
  let constructor = "triton::createTx81MemrefToLLVMPass()";
  let options = [];
}

#endif
