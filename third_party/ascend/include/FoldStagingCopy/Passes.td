/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025.
 * Licensed under the MIT license.
 */

#ifndef FOLD_STAGING_COPY_CONVERSION_PASSES
#define FOLD_STAGING_COPY_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def FoldStagingCopy : Pass<"fold-staging-copy", "mlir::ModuleOp"> {
  let summary = "Fold redundant staging memref.alloc + memref.copy pairs "
                "produced by TritonToLinalgIncubated's tt.load lowering.";
  let description = [{
    TritonToLinalgIncubated lowers a tt.load on a block-pointer into:
      %stage = memref.alloc() : memref<SxE>
      memref.copy %src(GM), %stage
      bufferization.to_tensor %stage
    When the resulting tensor path subsequently feeds a memref.copy
    whose destination has an explicit address space (e.g. cbuf / L1),
    the intermediate staging buffer is redundant.  This pass folds the
    chain into:
      memref.copy %src(GM), %dst(#space)            // direct DMA
      bufferization.to_tensor %dst (after memory_space_cast)
    eliminating one alloc, one copy, and the risk of the staging alloc
    being later mis-inferred as L1 by InferHIVMMemScope.
  }];
  let constructor = "mlir::triton::createFoldStagingCopyPass()";
  let dependentDialects = [
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif // FOLD_STAGING_COPY_CONVERSION_PASSES
