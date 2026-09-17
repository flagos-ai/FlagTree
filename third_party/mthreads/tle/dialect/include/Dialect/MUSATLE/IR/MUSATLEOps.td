#ifndef MUSA_TLE_OPS
#define MUSA_TLE_OPS

#ifdef __TLE__
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MUSATLEDialect.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
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

def MUSATLE_BarrierAllocOp : MUSATLE_Op<"barrier.alloc"> {
  let summary = "declare an mthreads hardware barrier id range";
  let arguments = (ins
    I32Attr:$num_barriers,
    I32Attr:$arrive_count,
    I32Attr:$init_polarity,
    OptionalAttr<I32Attr>:$expect_bytes
  );
  let results = (outs I32:$base_id);
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def MUSATLE_BarrierIndexOp : MUSATLE_Op<"barrier.index", [Pure]> {
  let summary = "select an mthreads hardware barrier id from a range";
  let arguments = (ins I32:$base_id, I32:$index);
  let results = (outs I32:$bar_id);
  let assemblyFormat = "$base_id `[` $index `]` attr-dict";
  let hasVerifier = 1;
}

def MUSATLE_BarrierWaitOp : MUSATLE_Op<"barrier.wait"> {
  let summary = "wait on an mthreads hardware barrier id";
  let arguments = (ins I32:$bar_id, I32:$phase);
  let assemblyFormat = "$bar_id `,` $phase attr-dict";
  let hasVerifier = 1;
}

def MUSATLE_BarrierArriveOp : MUSATLE_Op<"barrier.arrive"> {
  let summary = "arrive on an mthreads hardware barrier id";
  let arguments = (ins
    I32:$bar_id,
    I32:$phase,
    I32Attr:$arrive_count
  );
  let assemblyFormat = "$bar_id `,` $phase attr-dict";
  let hasVerifier = 1;
}

def MUSATLE_PipeCreateOp : MUSATLE_Op<"pipe.create"> {
  let summary = "create a typed MUSA TLE pipe";
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

def MUSATLE_PipeWriterAcquireOp : MUSATLE_Op<"pipe.writer_acquire"> {
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

def MUSATLE_PipeWriterCommitOp : MUSATLE_Op<"pipe.writer_commit"> {
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

def MUSATLE_PipeWriterCloseOp : MUSATLE_Op<"pipe.writer_close"> {
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

def MUSATLE_PipeReaderWaitOp : MUSATLE_Op<"pipe.reader_wait"> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I1:$phase,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names,
    OptionalAttr<StrAttr>:$reader_name,
    OptionalAttr<ArrayAttr>:$reader_fields
  );
  let results = (outs I1:$is_closed);
  let assemblyFormat = "$fields `[` $stage `,` $phase `]` attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def MUSATLE_PipeReaderReleaseOp : MUSATLE_Op<"pipe.reader_release", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names,
    OptionalAttr<StrAttr>:$reader_name,
    OptionalAttr<ArrayAttr>:$reader_fields
  );
  let assemblyFormat = "$fields `[` $stage `]` attr-dict `:` type($fields)";
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

def MUSATLE_SetLayoutOp : MUSATLE_Op<"set_layout", [
    Pure,
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType
]> {
  let summary = "Attach a requested distributed encoding to a tensor value";
  let description = [{
    Carries a user-requested TritonGPU distributed encoding through the
    mthreads TLE frontend. The op is lowered during Triton-to-TritonGPU
    conversion after encoding propagation has selected legal layouts for
    dependent values.
  }];
  let arguments = (ins TT_Tensor:$src, AnyAttr:$target_encoding);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";
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

def MUSATLE_SqmmaOp : MUSATLE_Op<"sqmma", [
    TypesMatchWith<"result type matches accumulator type", "d", "c", "$_self">
  ]> {
  let summary = "MUSA TLE asynchronous SQMMA";
  let arguments = (ins
    TTG_MemDescType:$a,
    TTG_MemDescType:$b,
    TT_FpIntTensor:$c,
    DefaultValuedAttr<TT_InputPrecisionAttr,
                      "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
    DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc,
    DefaultValuedAttr<BoolAttr, "true">:$isAsync
  );
  let results = (outs TT_FpIntTensor:$d);
  let assemblyFormat = [{
    $a `,` $b `,` $c attr-dict `:` qualified(type($a)) `*`
    qualified(type($b)) `,` qualified(type($c)) `->` qualified(type($d))
  }];
  let hasVerifier = 1;
}

def MUSATLE_SqmmaWaitOp : MUSATLE_Op<"sqmma_wait", [
    TypesMatchWith<"result type matches input type", "output", "input", "$_self">
  ]> {
  let summary = "Wait for a MUSA TLE asynchronous SQMMA";
  let arguments = (ins TT_FpIntTensor:$input, I32Attr:$pendings);
  let results = (outs TT_FpIntTensor:$output);
  let assemblyFormat = [{
    $input attr-dict `:` qualified(type($input)) `->` qualified(type($output))
  }];
  let hasVerifier = 1;
}
#endif // __TLE__

#endif // MUSA_TLE_OPS
