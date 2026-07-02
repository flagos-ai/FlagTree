#ifndef LLVMXPU_PASSES
#define LLVMXPU_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Functionality Pass
//===----------------------------------------------------------------------===//

def LLVMXPUMfenceCheck : Pass<"llvmxpu-mfence-check", "mlir::LLVM::LLVMFuncOp"> {
  let summary = "Insert Mfence To Prevent WAR Conflicts";
  let description = [{
    Insert Mfence To Prevent WAR Conflicts.
  }];
  let dependentDialects = ["mlir::LLVM::XPU::LLVMXPUDialect"];
}

#endif // LLVMXPU_PASSES
