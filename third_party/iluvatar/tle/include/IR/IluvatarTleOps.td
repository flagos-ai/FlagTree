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

def IluvatarTle_SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

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

def IluvatarTle_ExclusiveCumsumOp
    : IluvatarTle_Op<"exclusive_cumsum", [Pure, SameOperandsAndResultEncoding]> {
  let summary = "compute exclusive prefix sum and total sum along an axis";
  let description = [{
    Compute an exclusive cumulative sum (`exclusive[i] = sum(src[:i])`, or the
    reverse-exclusive variant when `reverse` is set) together with the total sum
    of `src`. Lowered to `tt.scan` + `tt.reduce` before TritonGPU codegen.
  }];
  let arguments = (ins TT_Tensor:$src, I32Attr:$axis, BoolAttr:$reverse);
  let results = (outs TT_Tensor:$exclusive, TT_Type:$total);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($exclusive) `,` type($total)";
  let hasVerifier = 1;
}

def IluvatarTle_InitBarrierOp : IluvatarTle_Op<"init_barrier"> {
  let summary = "Initialize a barrier in the given shared memory allocation.";
  let arguments = (ins
    Arg<TTG_MemDescType, "", [MemWrite<IluvatarTle_SharedMemory>]>:$alloc,
    I32Attr:$count
  );
  let assemblyFormat = "$alloc `,` $count attr-dict `:` qualified(type($alloc))";
  let hasVerifier = 1;
}

def IluvatarTle_WaitBarrierOp : IluvatarTle_Op<"wait_barrier"> {
  let summary = "wait until a software mbarrier phase completes (ivcore11)";
  let arguments = (ins
    Arg<TTG_MemDescType, "",
        [MemRead<IluvatarTle_SharedMemory>, MemWrite<IluvatarTle_SharedMemory>]>:$alloc,
    I32:$phase
  );
  let assemblyFormat = "$alloc `,` $phase attr-dict `:` qualified(type($alloc))";
  let hasVerifier = 1;
}

def IluvatarTle_ArriveBarrierOp : IluvatarTle_Op<"arrive_barrier"> {
  let summary = "arrive at a software mbarrier (ivcore11)";
  let arguments = (ins
    Arg<TTG_MemDescType, "",
        [MemRead<IluvatarTle_SharedMemory>, MemWrite<IluvatarTle_SharedMemory>]>:$alloc,
    I32Attr:$count
  );
  let assemblyFormat = "$alloc `,` $count attr-dict `:` qualified(type($alloc))";
  let hasVerifier = 1;
}

def IluvatarTle_PipeCreateOp : IluvatarTle_Op<"pipe.create"> {
  let summary = "create a typed TLE GPU pipe";
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names,
    OptionalAttr<ArrayAttr>:$readers,
    OptionalAttr<BoolAttr>:$one_shot
  );
  let assemblyFormat = "$fields attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def IluvatarTle_PipeWriterAcquireOp : IluvatarTle_Op<"pipe.writer_acquire"> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I1:$phase,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$fields `[` $stage `,` $phase `]` attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def IluvatarTle_PipeWriterCommitOp : IluvatarTle_Op<"pipe.writer_commit"> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$fields `[` $stage `]` attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def IluvatarTle_PipeWriterCloseOp : IluvatarTle_Op<"pipe.writer_close"> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I1:$phase,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$fields `[` $stage `,` $phase `]` attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def IluvatarTle_PipeReaderWaitOp : IluvatarTle_Op<"pipe.reader_wait"> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I1:$phase,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names,
    OptionalAttr<StrAttr>:$reader_name
  );
  let results = (outs I1:$is_closed);
  let assemblyFormat = "$fields `[` $stage `,` $phase `]` attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def IluvatarTle_PipeReaderReleaseOp : IluvatarTle_Op<"pipe.reader_release", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names,
    OptionalAttr<StrAttr>:$reader_name
  );
  let assemblyFormat = "$fields `[` $stage `]` attr-dict `:` type($fields)";
  let hasVerifier = 1;
}
#endif // __ILUVATAR_TLE__

#endif // ILUVATAR_TLE_OPS
