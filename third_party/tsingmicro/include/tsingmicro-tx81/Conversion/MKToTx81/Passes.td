//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//

#ifndef MK_TO_TX81_CONVERSION_PASSES
#define MK_TO_TX81_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def MKToTx81 : Pass<"mk-to-tx81", "mlir::ModuleOp"> {
  let summary = "Convert magic kernel operations into TsingMicro Tx81 operations";
  let constructor = "triton::createMKToTx81Pass()";
}

#endif
