//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_LINALG_TILING_PASSES
#define TRITON_CONVERSION_LINALG_TILING_PASSES

include "mlir/Pass/PassBase.td"

def LinalgTiling : Pass<"linalg-tiling", "mlir::ModuleOp"> {
  let summary = "Apply tiling transformation to linalg operations";
  let description = [{
    This pass applies tiling transformation to linalg operations.
    It tiles and fuses linalg operations to improve data locality and parallelism.
  }];
  let constructor = "triton::createLinalgTilingPass()";
}

#endif // TRITON_CONVERSION_LINALG_TILING_PASSES
