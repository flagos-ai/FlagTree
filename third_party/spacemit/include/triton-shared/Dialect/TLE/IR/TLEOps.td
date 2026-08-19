#ifndef TLE_OPS
#define TLE_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton-shared/Dialect/TLE/IR/TLEDialect.td"

class TLE_Op<string mnemonic, list<Trait> traits = []> :
    Op<TLE_Dialect, mnemonic, traits>;

// ============================================================================
// ExtractTileOp
// ============================================================================
def TLE_ExtractTileOp : TLE_Op<"extract_tile", [Pure]> {
  let summary = "Extract a tile from a tensor at a given tile index";

  let description = [{
    Extract a contiguous tile from a tensor based on a linearized tile index.
    The tile grid is computed from source shape / tile_shape (row-major order).

    Index can be:
      - Static (arith.constant): full compile-time validation
      - Dynamic (runtime value): only compile-time-checkable constraints verified

    Requirements:
      1. Source shape must be divisible by tile_shape in each dimension
      2. Result shape must equal tile_shape
      3. Element types must match
  }];

  let arguments = (ins TT_Tensor:$src, TT_IntLike:$index);
  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Value":$src, "Value":$index,
        "ArrayRef<int64_t>":$tileShape)>
  ];

  let assemblyFormat = [{
    $src `[` $index `]` attr-dict `:` qualified(type($src)) `,`
    qualified(type($index)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

// ============================================================================
// InsertTileOp
// ============================================================================
def TLE_InsertTileOp : TLE_Op<"insert_tile", [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Insert a tile into a tensor at a given tile index";

  let description = [{
    Insert a tile into a source tensor at the specified linearized tile index.
    Returns a new tensor with the tile values written at the target position.
    This is a functional (non-in-place) operation.

    The tile grid is computed from source shape / tile shape (row-major order).
  }];

  let arguments = (ins TT_Tensor:$src, TT_Tensor:$tile, TT_IntLike:$index);
  let results = (outs TT_Tensor:$result);

  let assemblyFormat = [{
    $src `[` $index `]` `=` $tile attr-dict `:` qualified(type($src)) `,`
    qualified(type($index)) `,` qualified(type($tile)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

#endif
