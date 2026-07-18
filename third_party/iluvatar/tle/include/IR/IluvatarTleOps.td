#ifndef ILUVATAR_TLE_OPS
#define ILUVATAR_TLE_OPS

#ifdef __ILUVATAR_TLE__
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "IluvatarTleDialect.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class IluvatarTle_Op<string mnemonic, list<Trait> traits = []>
    : Op<IluvatarTle_Dialect, mnemonic,
         !listconcat(traits, [VerifyTensorLayoutsTrait])> {}

def IluvatarTle_ExtractTileOp : IluvatarTle_Op<"extract_tile", [Pure]> {
  let arguments = (ins TT_Tensor:$src, TT_IntLike:$index);
  let results = (outs TT_Tensor:$result);
  let builders = [OpBuilder<(ins "Value":$src, "Value":$index,
      "ArrayRef<int64_t>":$tileShape)>];
  let assemblyFormat = [{
    $src `[` $index `]` attr-dict `:` qualified(type($src)) `,` qualified(type($index)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def IluvatarTle_InsertTileOp
    : IluvatarTle_Op<"insert_tile", [Pure,
                                     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let arguments = (ins TT_Tensor:$src, TT_Tensor:$tile, TT_IntLike:$index);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $src `[` $index `]` `=` $tile attr-dict `:` qualified(type($src)) `,` qualified(type($index)) `,` qualified(type($tile)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def IluvatarTle_LocalPointerResultType : AnyTypeOf<[TT_Tensor, TT_Ptr]>;
def IluvatarTle_LocalPointerIndexType : AnyTypeOf<[TT_Tensor, TT_Int]>;

def IluvatarTle_LocalPointersOp : IluvatarTle_Op<"local_pointers", [Pure]> {
  let arguments = (ins TTG_MemDescType:$src,
      Variadic<IluvatarTle_LocalPointerIndexType>:$indices);
  let results = (outs IluvatarTle_LocalPointerResultType:$result);
  let hasVerifier = 1;
}
#endif // __ILUVATAR_TLE__

#endif // ILUVATAR_TLE_OPS
