#ifndef MUSA_TLE_OPS
#define MUSA_TLE_OPS

#ifdef __TLE__
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MUSATLEDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class MUSATLE_Op<string mnemonic, list<Trait> traits = []>
    : Op<MUSATLE_Dialect, mnemonic,
         !listconcat(traits, [VerifyTensorLayoutsTrait])> {}

def MUSATLE_LocalPointerResultType : AnyTypeOf<[TT_Tensor, TT_Ptr]>;
def MUSATLE_LocalPointerIndexType : AnyTypeOf<[TT_Tensor, TT_Int]>;

def MUSATLE_LocalPointersOp : MUSATLE_Op<"local_pointers", [Pure]> {
  let arguments = (ins TTG_MemDescType:$src,
      Variadic<MUSATLE_LocalPointerIndexType>:$indices);
  let results = (outs MUSATLE_LocalPointerResultType:$result);
  let hasVerifier = 1;
}

def MUSATLE_ExtractTileOp : MUSATLE_Op<"extract_tile", [Pure]> {
  let arguments = (ins TT_Tensor:$src, TT_IntLike:$index,
      DenseI64ArrayAttr:$tile_shape);
  let results = (outs TT_Tensor:$result);
  let builders = [OpBuilder<(ins "Value":$src, "Value":$index,
      "ArrayRef<int64_t>":$tileShape)>];
  let assemblyFormat = [{
    $src `[` $index `]` attr-dict `:` qualified(type($src)) `,` qualified(type($index)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MUSATLE_InsertTileOp
    : MUSATLE_Op<"insert_tile", [Pure,
                                 DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins TT_Tensor:$src, TT_Tensor:$tile, TT_IntLike:$index,
      DenseI64ArrayAttr:$tile_shape);
  let results = (outs TT_Tensor:$result);
  let builders = [OpBuilder<(ins "Value":$src, "Value":$tile,
      "Value":$index)>];
  let assemblyFormat = [{
    $src `[` $index `]` `=` $tile attr-dict `:` qualified(type($src)) `,` qualified(type($index)) `,` qualified(type($tile)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MUSATLE_ExclusiveCumsumOp
    : MUSATLE_Op<"exclusive_cumsum",
                 [Pure, SameOperandsAndResultEncoding]> {
  let arguments = (ins TT_Tensor:$src, I32Attr:$axis, BoolAttr:$reverse);
  let results = (outs TT_Tensor:$exclusive, TT_Type:$total);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($exclusive) `,` type($total)";
  let hasVerifier = 1;
}
#endif // __TLE__

#endif // MUSA_TLE_OPS
