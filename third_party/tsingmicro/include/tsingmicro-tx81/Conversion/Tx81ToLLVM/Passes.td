//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//

#ifndef TX81_TO_LLVM_CONVERSION_PASSES
#define TX81_TO_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"


def Tx81ToLLVM : Pass<"tx81-to-llvm", "ModuleOp"> {
  let summary = "Convert Tx81 dialect to LLVM dialect";
  let description = [{
    This pass converts operations in the Tx81 dialect to the LLVM IR dialect.

    It handles the conversion of Tx81-specific operations like tx.rdma, tx.wdma,
    tx.gemm etc to appropriate LLVM calls to the Tx81 runtime library.

    The pass also relies on existing conversion patterns for standard dialects
    like arith, func, memref, etc.
  }];

  let constructor = "triton::createTx81ToLLVMPass()";

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect",
    "mlir::scf::SCFDialect",
    "tx::Tx81Dialect"
  ];
}

#endif
