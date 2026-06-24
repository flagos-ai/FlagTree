/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TILE_IR_TO_HIVM_CONVERSION_PASSES
#define TILE_IR_TO_HIVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TileIRToHIVM : Pass<"tileir-to-hivm", "mlir::ModuleOp"> {
  let summary = "Convert TileIR dialect to HIVM dialect";
  let constructor = "triton::createTileIRToHIVMPass()";
  let dependentDialects = [
    "hivm::HIVMDialect",
    "triton::tile::TileIRDialect"
  ];
}

#endif // TILE_IR_TO_HIVM_CONVERSION_PASSES