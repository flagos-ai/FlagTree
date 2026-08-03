#ifndef MK_TRANSFORMS_PASSES
#define MK_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MaterializeStridedLinalgInputs
    : Pass<"materialize-strided-linalg-inputs", "mlir::ModuleOp"> {
  let summary = "Materialize non-contiguous subview inputs of compute linalg.generic ops";

  let description = [{
    This pass runs after OneShotBufferize.

    Some TX device compute ops do not support strided memref operands. After
    bufferization, tensor.extract_slice may become memref.subview. If a
    linalg.generic compute op consumes such a non-contiguous subview directly,
    later lowering may drop or ignore the subview strides and read incorrect
    values.

    This pass finds compute linalg.generic ops whose input operands are
    non-contiguous memref.subview values. For each such input, it creates a
    contiguous memref.alloc, inserts memref.copy from the subview into the
    allocation, and rewrites the linalg.generic input to use the contiguous
    buffer.

    Example:

      %sv = memref.subview %arg0[...] :
          memref<1x4x2xf32> to memref<1x4xf32, strided<[8, 2], offset: 1>>

      linalg.generic ins(%sv : memref<1x4xf32, strided<[8, 2], offset: 1>>) ...

    becomes:

      %tmp = memref.alloc() : memref<1x4xf32>
      memref.copy %sv, %tmp :
          memref<1x4xf32, strided<[8, 2], offset: 1>> to memref<1x4xf32>

      linalg.generic ins(%tmp : memref<1x4xf32>) ...
  }];

  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect"
  ];

  let constructor = "triton::createMaterializeStridedLinalgInputsPass()";
}

#endif // MK_TRANSFORMS_PASSES