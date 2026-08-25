#ifndef TLE_RAW_OPS
#define TLE_RAW_OPS

include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "tle/dialect/include/IR/TleDialect.td"
include "tle/dialect/include/IR/TleAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class Tle_Op<string mnemonic, list<Trait> traits = []>
    : Op<Tle_Dialect, mnemonic,
         !listconcat(traits, [VerifyTensorLayoutsTrait])> {}

// Begin TLE-Lite
def Tle_ExtractTileOp : Tle_Op<"extract_tile", [Pure]> {
  let summary = "Extract a tile from a tensor with static offsets";

  let description = [{
    Extract a contiguous tile from a tensor based on static offsets.
    This is a zero-cost operation at runtime - it only reorganizes
    register values within each thread, with no inter-thread communication.

    Requirements:
      1. Offsets must be multiples of CTA tile sizes
      2. Tile shape must be multiples of CTA tile sizes
      3. Source and destination layouts must match at CTA tile level
  }];

  let arguments = (ins TT_Tensor:$src, TT_IntLike:$index);

  let results = (outs TT_Tensor:$result);
  let builders = [OpBuilder<(ins "Value":$src, "Value":$index,
      "ArrayRef<int64_t>":$tileShape)>];
  let assemblyFormat = [{
    $src `[` $index `]` attr-dict `:` qualified(type($src)) `,` qualified(type($index)) `->` qualified(type($result))
  }];

  let hasVerifier = 1;
}

def Tle_InsertTileOp
    : Tle_Op<"insert_tile", [Pure,
                             DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let arguments = (ins TT_Tensor:$src, TT_Tensor:$tile, TT_IntLike:$index);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $src `[` $index `]` `=` $tile attr-dict `:` qualified(type($src)) `,` qualified(type($index)) `,` qualified(type($tile)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Tle_EncodingOp : Tle_Op<"encoding", [
    Pure,
    SameOperandsAndResultShape,
    SameOperandsAndResultElementType
]> {
  let summary = "Attach a requested distributed encoding to a tensor value";

  let description = [{
    Carries a user-requested TritonGPU distributed encoding through the TLE
    frontend. The op is lowered during Triton-to-TritonGPU conversion, after
    encoding propagation has selected legal layouts for dependent values.
  }];

  let arguments = (ins TT_Tensor:$src, AnyAttr:$target_encoding);

  let results = (outs TT_Tensor:$result);

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";
}

// End TLE-Lite

// Begin TLE-Raw

def Tle_TensorType : AnyTypeOf<[TT_Type, TTG_MemDescType]>;
def Tle_ArgType : AnyTypeOf<[Tle_TensorType, LLVMPointerType, LLVMStructType]>;
def Tle_LocalPointerResultType : AnyTypeOf<[TT_Tensor, TT_Ptr]>;
def Tle_LocalPointerIndexType : AnyTypeOf<[TT_Tensor, TT_Int]>;

def Tle_LocalPointersOp : Tle_Op<"local_pointers", [Pure]> {
  let arguments = (ins TTG_MemDescType:$src,
      Variadic<Tle_LocalPointerIndexType>:$indices);
  let results = (outs Tle_LocalPointerResultType:$result);
  let hasVerifier = 1;
}

def Tle_MemDescWGMMAViewOp : Tle_Op<"memdesc_wgmma_view", [Pure,
                                                           MemDescViewTrait,
                                                           SameOperandsAndResultElementType]> {
  let summary = "create a WGMMA descriptor view of a shared memory descriptor";

  let description = [{
    This operation returns a descriptor-only view of an existing shared memory
    allocation for WGMMA consumers. It does not allocate or move shared memory.

    The result type carries the logical matrix shape and WGMMA shared encoding
    required by the consumer. The source descriptor provides the underlying
    storage. The order attribute describes how logical offsets are mapped from
    the source view to the result view.
  }];

  let arguments = (
    ins TTG_MemDescType:$src,
    DenseI32ArrayAttr:$order
  );

  let results = (outs TTG_MemDescType:$result);

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";

  let hasVerifier = 1;
}

def Tle_MemDescAliasOp : Tle_Op<"memdesc_alias", [Pure, MemDescViewTrait]> {
  let summary = "create a typed view that aliases an existing shared memory descriptor";

  let description = [{
    This operation returns a descriptor-only view into an existing shared memory
    descriptor. It does not allocate or move shared memory.

    The result type carries the logical shape, element type, and encoding used
    by consumers. The source descriptor provides the backing storage. The
    offset_bytes attribute is a static byte offset from the source descriptor's
    current view base.
  }];

  let arguments = (
    ins TTG_MemDescType:$src,
    I64Attr:$offset_bytes
  );

  let results = (outs TTG_MemDescType:$result);

  let assemblyFormat = "$src attr-dict `:` qualified(type($src)) `->` qualified(type($result))";

  let hasVerifier = 1;
}

def Tle_PipeCreateOp : Tle_Op<"pipe.create"> {
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
  let results = (outs I32:$identity);
  let assemblyFormat = "($fields^)? attr-dict (`:` type($fields)^)?";
  let hasVerifier = 1;
}

def Tle_PipeWriterAcquireOp : Tle_Op<"pipe.writer_acquire"> {
  let arguments = (ins
    I32:$identity,
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I1:$phase,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$identity `,` $fields `[` $stage `,` $phase `]` attr-dict (`:` type($fields)^)?";
  let hasVerifier = 1;
}

def Tle_PipeWriterCommitOp : Tle_Op<"pipe.writer_commit"> {
  let arguments = (ins
    I32:$identity,
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$identity `,` $fields `[` $stage `]` attr-dict (`:` type($fields)^)?";
  let hasVerifier = 1;
}

def Tle_PipeWriterCloseOp : Tle_Op<"pipe.writer_close"> {
  let arguments = (ins
    I32:$identity,
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I1:$phase,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$identity `,` $fields `[` $stage `,` $phase `]` attr-dict (`:` type($fields)^)?";
  let hasVerifier = 1;
}

def Tle_PipeReaderWaitOp : Tle_Op<"pipe.reader_wait"> {
  let arguments = (ins
    I32:$identity,
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
  let assemblyFormat = "$identity `,` $fields `[` $stage `,` $phase `]` attr-dict (`:` type($fields)^)?";
  let hasVerifier = 1;
}

def Tle_PipeReaderReleaseOp : Tle_Op<"pipe.reader_release", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let arguments = (ins
    I32:$identity,
    Variadic<TTG_MemDescType>:$fields,
    I32:$stage,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names,
    OptionalAttr<StrAttr>:$reader_name
  );
  let assemblyFormat = "$identity `,` $fields `[` $stage `]` attr-dict (`:` type($fields)^)?";
  let hasVerifier = 1;
}

def Tle_PipeDrainOp : Tle_Op<"pipe.drain", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "rendezvous all pipe participants before whole-storage reuse";
  let description = [{
    `tle.pipe.drain` marks a one-shot lifecycle boundary for a whole pipe. The
    op is emitted by the writer task after `pipe.writer_close` and by every
    reader task after its final `pipe.reader_release`. Lowering verifies that
    all required participants reach the drain and emits a dedicated named
    barrier before the pipe backing storage may be reused through aliases.
  }];
  let arguments = (ins
    I32:$identity,
    Variadic<TTG_MemDescType>:$fields,
    I32Attr:$capacity,
    StrAttr:$scope,
    OptionalAttr<StrAttr>:$pipe_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$identity `,` $fields attr-dict (`:` type($fields)^)?";
  let hasVerifier = 1;
}

def Tle_PipeCallBeginOp : Tle_Op<"pipe.call_begin"> {
  let summary = "mark the beginning of a temporarily inlined pipe helper call";
  let arguments = (ins
    Variadic<AnyType>:$arguments,
    StrAttr:$callee,
    I64Attr:$call_id
  );
  let results = (outs Variadic<AnyType>:$aliases);
  let assemblyFormat = "$arguments attr-dict `:` functional-type($arguments, $aliases)";
  let hasVerifier = 1;
}

def Tle_PipeCallEndOp : Tle_Op<"pipe.call_end"> {
  let summary = "mark the end of a temporarily inlined pipe helper call";
  let arguments = (ins StrAttr:$callee, I64Attr:$call_id);
  let assemblyFormat = "attr-dict";
}

def Tle_TaskGridCreateOp : Tle_Op<"task_grid.create"> {
  let summary = "create a typed TLE task-grid dependency edge";
  let arguments = (ins
    Variadic<Tle_TensorType>:$fields,
    StrAttr:$scope,
    StrAttr:$grid_name,
    ArrayAttr:$field_names,
    OptionalAttr<DenseI64ArrayAttr>:$shape
  );
  let assemblyFormat = "$fields attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def Tle_TaskGridTileIdOp : Tle_Op<"task_grid.tile_id"> {
  let arguments = (ins
    Variadic<Tle_TensorType>:$fields,
    StrAttr:$scope,
    StrAttr:$grid_name,
    ArrayAttr:$field_names,
    OptionalAttr<DenseI64ArrayAttr>:$shape
  );
  let results = (outs Variadic<I32>:$ids);
  let hasVerifier = 1;
}

def Tle_TaskGridCommitOp : Tle_Op<"task_grid.commit"> {
  let arguments = (ins
    Variadic<Tle_TensorType>:$fields,
    StrAttr:$scope,
    StrAttr:$grid_name,
    ArrayAttr:$field_names
  );
  let assemblyFormat = "$fields attr-dict `:` type($fields)";
  let hasVerifier = 1;
}

def Tle_TaskDeclareOp : Tle_Op<"task.declare"> {
  let summary = "declare a tile task domain and task-grid dependency maps";
  let description = [{
    Marker operation that attaches task metadata to the enclosing tt.func.
    The task body remains the enclosing function body for this milestone. The
    reads and writes arrays contain dictionaries with:

      - grid: string task-grid name
      - map: affine map from task domain coordinates to non-wildcard grid dims
      - wildcard_dims: optional i64 dense array for read-all grid dimensions

    Wildcard dimensions are allowed only on reads.
  }];
  let arguments = (ins
    StrAttr:$task_name,
    OptionalAttr<FlatSymbolRefAttr>:$callee,
    DenseI64ArrayAttr:$domain_shape,
    ArrayAttr:$reads,
    ArrayAttr:$writes
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tle_ExclusiveCumsumOp : Tle_Op<"exclusive_cumsum", [Pure, SameOperandsAndResultEncoding]> {
  let arguments = (ins TT_Tensor:$src, I32Attr:$axis, BoolAttr:$reverse);
  let results = (outs TT_Tensor:$exclusive, TT_Type:$total);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($exclusive) `,` type($total)";
  let hasVerifier = 1;
}

def Tle_WGMMASharedOperandFenceOp : Tle_Op<"wgmma_shared_operand_fence"> {
  let arguments = (ins
    Variadic<TTG_MemDescType>:$deps,
    BoolAttr:$bCluster
  );

  let summary = "fence shared-memory operands before WGMMA async-proxy reads";

  let description = [{
    Orders prior generic-proxy shared-memory writes to the listed WGMMA shared
    operands before a following WGMMA reads them through the async proxy.
  }];

  let assemblyFormat = "$deps attr-dict `:` type($deps)";
  let hasVerifier = 1;
}

def Tle_TMAStoreCommitGroupOp
    : Tle_Op<"tma_store.commit_group", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "commit the current TMA store bulk group";

  let description = [{
    Commits outstanding shared-to-global TMA store messages issued by this
    agent into the current bulk group.  This op makes TMA store group
    boundaries explicit in TLE IR so scheduling passes can group adjacent
    stores and place waits at the latest safe point.
  }];

  let assemblyFormat = "attr-dict";
}

def Tle_InitBarrierGroupOp
    : Tle_Op<"init_barrier_group", [MemoryEffects<[MemWrite]>]> {
  let summary = "initialize a static group of shared-memory barriers";

  let description = [{
    Initializes barriers at compiler-assigned shared-memory byte offsets. The
    offset and participant-count arrays are parallel. Backends distribute the
    work over the first `worker_count` physical CTA threads while keeping each
    participant count as a target instruction immediate.

    This op is created after shared-memory allocation and is intended for
    kernels with enough independent barriers to amortize table dispatch.
  }];

  let arguments = (ins
    DenseI32ArrayAttr:$offsets,
    DenseI32ArrayAttr:$counts,
    I32Attr:$worker_count
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tle_DistributedBarrierOp : Tle_Op<"distributed_barrier",
                                      [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins
    OptionalAttr<StrAttr>:$group_kind,
    OptionalAttr<I32Attr>:$group_rank,
    OptionalAttr<DenseI32ArrayAttr>:$group_shape,
    OptionalAttr<DenseI32ArrayAttr>:$group_axes,
    OptionalAttr<DenseI32ArrayAttr>:$group_mask,
    OptionalAttr<DenseI32ArrayAttr>:$group_domain_shape
  );
  let assemblyFormat = "attr-dict";
  let hasVerifier = 1;
}

def Tle_ClusterCTAIdOp : Tle_Op<"cluster_cta_id", [Pure]> {
  let summary = "return the hardware CTA rank inside the current cluster";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict";
}

def Tle_RemotePointersOp : Tle_Op<"remote_pointers", [Pure]> {
  let arguments = (ins Tle_LocalPointerResultType:$src, TT_Int:$shard_id);
  let results = (outs Tle_LocalPointerResultType:$result);
  let hasVerifier = 1;
}

def Tle_DSLRegionOp : Tle_Op<"dsl_region", [IsolatedFromAbove, MemDescViewTrait,
                                            RecursiveMemoryEffects]> {
  let arguments = (ins
    Variadic<Tle_ArgType>:$inputs
  );
  let results = (outs Variadic<Tle_ArgType>:$outputs);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Tle_YieldOp : Tle_Op<"yield", [Pure, Terminator, ReturnLike,
                                   HasParent<"DSLRegionOp">]> {
  let arguments = (ins Variadic<Tle_ArgType>:$inputs);
  let assemblyFormat = "attr-dict ($inputs^ `:` type($inputs))?";
}

def Tle_ExtractAllocatedPtrOp
    : Tle_Op<"extract_allocated_ptr", [Pure]> {
  let arguments = (ins Tle_ArgType:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_ExtractAlignedPtrOp
    : Tle_Op<"extract_aligned_ptr", [Pure]> {
  let arguments = (ins Tle_ArgType:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_ExtractOffsetOp
    : Tle_Op<"extract_offset", [Pure]> {
  let arguments = (ins Tle_TensorType:$input);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_ExtractSizesOp
    : Tle_Op<"extract_sizes", [Pure]> {
  let arguments = (ins Tle_TensorType:$input);
  let results = (outs Variadic<I64>:$sizes);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($sizes)";
  let builders = [OpBuilder<(ins "size_t":$num, "Value":$input)>];
}

def Tle_ExtractStridesOp
    : Tle_Op<"extract_strides", [Pure]> {
  let arguments = (ins Tle_TensorType:$input);
  let results = (outs Variadic<I64>:$strides);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($strides)";
  let builders = [OpBuilder<(ins "size_t":$num, "Value":$input)>];
}

def Tle_ExtractPtrOp : Tle_Op<"extract_ptr", [Pure]> {
  let arguments = (ins TT_Ptr:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_PackOp
    : Tle_Op<"pack", [MemDescViewTrait, Pure]> {
  let arguments = (ins LLVMStructType:$input);
  let results = (outs Tle_TensorType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let hasVerifier = 1;
}

#endif
