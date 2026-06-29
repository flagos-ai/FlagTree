/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * Licensed under the MIT license.
 */

#ifndef ERASE_LINALG_CASTS_CONVERSION_PASSES
#define ERASE_LINALG_CASTS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def EraseLinalgCasts : Pass<"erase-linalg-casts", "mlir::ModuleOp"> {
  let summary = "Lower the unrealized_conversion_cast ops left by TileIRToHIVM "
                "into legal MLIR bridges (bufferization.to_tensor / "
                "bufferization.to_memref / tt.load), so that the subsequent "
                "triton-to-linalg-incubated pass produces clean linalg-dialect IR.";
  let description = [{
    The TileIRToHIVM pass converts:
      * tile.alloc          -> memref.alloc(addrspace)
      * tile.to_tensor      -> unrealized_conversion_cast (memref -> tensor)
      * tile.copy %ptr,%dst -> memref.copy with
                               unrealized_conversion_cast (!tt.ptr<tensor<>> -> memref)
    Those unrealized casts confuse the triton-to-linalg-incubated pass, which
    leaves orphan "unresolved materialization" errors.  This pass rewrites the
    two cast patterns into legitimate MLIR ops:
      Pattern A  unrealized_conversion_cast %ptr : !tt.ptr<tensor<TxS,E>>
                                                  to memref<TxS,E>
        replaced with:  %t = tt.load %ptr
                        %m = bufferization.to_memref %t
      Pattern B  unrealized_conversion_cast %m   : memref<TxS,E,space>
                                                  to tensor<TxS,E>
        replaced with:  %mg = memref.memory_space_cast %m (drop space)
                        %t  = bufferization.to_tensor %mg restrict
    Any other one-to-one unrealized casts that form A->B->A chains are folded
    away by ReconcileUnrealizedCasts at the end.
  }];
  let constructor = "mlir::triton::createEraseLinalgCastsPass()";
  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::triton::TritonDialect",
  ];
}

#endif // ERASE_LINALG_CASTS_CONVERSION_PASSES
