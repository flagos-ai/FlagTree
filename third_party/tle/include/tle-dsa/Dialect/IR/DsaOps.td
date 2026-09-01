//===- DsaOps.td - TLE DSA dialect ops --------------------*- tablegen -*-===//
//
// Template ops for TLE-Struct style DSA extensions.
//
//===----------------------------------------------------------------------===//

#ifndef TLE_DSA_OPS
#define TLE_DSA_OPS

include "tle-dsa/Dialect/IR/DsaDialect.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

//===----------------------------------------------------------------------===//
// dsa.alloc
//===----------------------------------------------------------------------===//

def Dsa_AllocOp : Dsa_Op<"alloc", [MemoryEffects<[MemAlloc]>]> {
  let summary = "Allocate a DSA local buffer";
  let arguments = (ins DenseI64ArrayAttr:$shape);
  let results = (outs AnyRankedOrUnrankedMemRef:$result);
  let assemblyFormat = [{
    $shape attr-dict `:` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// dsa.copy
//===----------------------------------------------------------------------===//

def Dsa_CopyOp : Dsa_Op<"copy",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Copy between DSA local buffers";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$src, AnyRankedOrUnrankedMemRef:$dst);
  let results = (outs);
  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `,` type($dst)
  }];
}

//===----------------------------------------------------------------------===//
// dsa.add / dsa.sub / dsa.mul / dsa.maximum / dsa.minimum / dsa.div
// Elementwise binary arithmetic on local memrefs (three-operand: out = lhs OP rhs).
//===----------------------------------------------------------------------===//

class Dsa_BinaryOp<string mnemonic> : Dsa_Op<mnemonic,
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins AnyRankedOrUnrankedMemRef:$lhs,
                       AnyRankedOrUnrankedMemRef:$rhs,
                       AnyRankedOrUnrankedMemRef:$out);
  let results = (outs);
  let assemblyFormat = [{
    $lhs `,` $rhs `,` $out attr-dict `:` type($lhs) `,` type($rhs) `,` type($out)
  }];
}

def Dsa_AddOp : Dsa_BinaryOp<"add"> {
  let summary = "elementwise add on local memrefs (out = lhs + rhs)";
}
def Dsa_SubOp : Dsa_BinaryOp<"sub"> {
  let summary = "elementwise subtract on local memrefs (out = lhs - rhs)";
}
def Dsa_MulOp : Dsa_BinaryOp<"mul"> {
  let summary = "elementwise multiply on local memrefs (out = lhs * rhs)";
}
def Dsa_MaximumOp : Dsa_BinaryOp<"maximum"> {
  let summary = "elementwise maximum on local memrefs (out = max(lhs, rhs))";
}
def Dsa_MinimumOp : Dsa_BinaryOp<"minimum"> {
  let summary = "elementwise minimum on local memrefs (out = min(lhs, rhs))";
}
def Dsa_DivOp : Dsa_BinaryOp<"div"> {
  let summary = "elementwise divide on local memrefs (out = lhs / rhs)";
}

def DsaLocalPointerResultType : AnyTypeOf<[TT_Tensor, TT_Ptr]>;
def DsaLocalPointerIndexType : AnyTypeOf<[TT_Tensor, TT_Int]>;
def DsaRemotePointerType : AnyTypeOf<[TT_Tensor, TT_Ptr]>;
def DsaRemoteShardIdType : AnyTypeOf<[TT_Tensor, TT_Int]>;

//===----------------------------------------------------------------------===//
// dsa.local_pointers / dsa.remote_pointers / dsa.distributed_barrier
// These are DSA-side structural counterparts of the official TLE distributed
// and pointer-building ops. They let the frontend move toward the same
// structured programming model without binding the design to GPU dialect names.
//===----------------------------------------------------------------------===//

def Dsa_LocalPointersOp : Dsa_Op<"local_pointers", [Pure]> {
  let arguments = (ins AnyRankedOrUnrankedMemRef:$src,
                   Variadic<DsaLocalPointerIndexType>:$indices);
  let results = (outs DsaLocalPointerResultType:$result);
}

def Dsa_DistributedBarrierOp : Dsa_Op<"distributed_barrier",
                                      [MemoryEffects<[MemRead, MemWrite]>]> {
  let arguments = (ins
    OptionalAttr<StrAttr>:$group_kind,
    OptionalAttr<I32Attr>:$group_rank,
    OptionalAttr<DenseI32ArrayAttr>:$group_shape,
    OptionalAttr<DenseI32ArrayAttr>:$group_axes,
    OptionalAttr<DenseI32ArrayAttr>:$group_mask
  );
  let assemblyFormat = "attr-dict";
}

def Dsa_RemotePointersOp : Dsa_Op<"remote_pointers", [Pure]> {
  let arguments = (ins DsaRemotePointerType:$src, DsaRemoteShardIdType:$shard_id,
                   OptionalAttr<DenseI32ArrayAttr>:$mesh_physical_ids,
                   OptionalAttr<DenseI32ArrayAttr>:$mesh_shape);
  let results = (outs DsaRemotePointerType:$result);
}

//===----------------------------------------------------------------------===//
// dsa.to_tensor / dsa.to_buffer
//
// Explicit buffer <-> tensor bridging:
//   - dsa.to_tensor: SPM buffer (memref) -> tl.tensor zero-copy view.
//     Lowered by TLEToMK to bufferization.to_tensor(memref, restrict, writable).
//   - dsa.to_buffer: tl.tensor -> copy into an existing SPM buffer (memref).
//     Lowered by TLEToMK to bufferization.to_memref + memref.copy.
//===----------------------------------------------------------------------===//

def Dsa_ToTensorOp : Dsa_Op<"to_tensor", [Pure]> {
  let summary = "Convert a DSA local buffer (memref) to a tl.tensor view";
  let arguments = (ins AnyRankedOrUnrankedMemRef:$src, BoolAttr:$writable);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $src attr-dict `:` type($src) `->` type($result)
  }];
}

def Dsa_ToBufferOp : Dsa_Op<"to_buffer", [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "Copy a tl.tensor value into an existing DSA local buffer (memref)";
  let arguments = (ins TT_Tensor:$src, AnyRankedOrUnrankedMemRef:$dst);
  let results = (outs);
  let assemblyFormat = [{
    $src `,` $dst attr-dict `:` type($src) `,` type($dst)
  }];
}

def DsaCumsumInputType : RankedTensorOf<[TT_Float, TT_Int]>;
def DsaCumsumTotalResultType : AnyTypeOf<[DsaCumsumInputType, TT_Float, TT_Int]>;

//===----------------------------------------------------------------------===//
// dsa.cumsum
//===----------------------------------------------------------------------===//

def Dsa_CumsumOp : Dsa_Op<"cumsum", [Pure]> {
  let summary = "exclusive cumsum marker lowered by Tsingmicro CRT";
  let arguments = (ins
    DsaCumsumInputType:$input,
    I32Attr:$axis,
    BoolAttr:$reverse,
    DenseI64ArrayAttr:$shape,
    I64Attr:$pad
  );
  let results = (outs
    DsaCumsumInputType:$exclusive,
    DsaCumsumTotalResultType:$total
  );
}

//===----------------------------------------------------------------------===//
// dsa.randgen for tsingmicro tx81
//===----------------------------------------------------------------------===//

def DsaRandGenSeedType : RankedTensorOf<[I64]>;
def DsaRandGenOutType : RankedTensorOf<[I64]>;

def Dsa_RandGenOp : Dsa_Op<"randgen",
    [MemoryEffects<[MemRead, MemWrite]>]> {
  let summary = "randgen for tsingmicro tx81";
  let description = [{
    Emits a single TX81 peri RandGen call. `seed0`/`seed1` are length-16 i64
    seed vectors; `byte_count` is the random output size in bytes and must be
    a multiple of 128. Results are `(out, seed0_out, seed1_out)`.
  }];
  let arguments = (ins
    DsaRandGenSeedType:$seed0,
    DsaRandGenSeedType:$seed1,
    I32Attr:$byte_count,
    I16Attr:$fmt
  );
  let results = (outs
    DsaRandGenOutType:$out,
    DsaRandGenSeedType:$seed0_out,
    DsaRandGenSeedType:$seed1_out
  );
  let assemblyFormat = [{
    $seed0 `,` $seed1 attr-dict `:` type($seed0) `,` type($seed1)
    `->` type($out) `,` type($seed0_out) `,` type($seed1_out)
  }];
}

//===----------------------------------------------------------------------===//
// dsa.bitcast
//===----------------------------------------------------------------------===//

def Dsa_BitcastOp : Dsa_Op<"bitcast", [Pure]> {
  let summary = "Same-nbytes type/shape reinterpret (vendor-neutral)";
  let description = [{
    Reinterpret `src` as `result` when both tensors have the same total bit
    size.  Element type and/or shape may change (e.g. `tensor<Nx i64>` →
    `tensor<(2N)x i32>`).  This is a pure view: no data movement.

    Backends lower this to a zero-cost alias when possible (e.g. Tsingmicro
    `mk.bitcast`); a portable fallback is `tensor.bitcast`.
  }];
  let arguments = (ins AnyRankedTensor:$src);
  let results = (outs AnyRankedTensor:$result);
  let assemblyFormat = [{
    $src attr-dict `:` type($src) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// dsa.extract_slice / dsa.insert_slice: strided tensor slice ops.
// Mixed static/dynamic per-dim offsets follow the tensor.extract_slice
// convention: `static_offsets` uses ShapedType::kDynamic as the sentinel for
// positions whose runtime value is carried by the variadic `offsets` operands.
// sizes/strides are always compile-time constants.
//===----------------------------------------------------------------------===//

def DsaSliceOffsetType : AnyTypeOf<[TT_Tensor, TT_Int]>;

def Dsa_ExtractSliceOp : Dsa_Op<"extract_slice", [Pure]> {
  let summary = "Extract a strided slice from a tl.tensor (mixed static/dynamic offsets, static sizes/strides)";
  let arguments = (ins TT_Tensor:$src,
                       Variadic<DsaSliceOffsetType>:$offsets,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides);
  let results = (outs TT_Tensor:$result);
}

def Dsa_InsertSliceOp : Dsa_Op<"insert_slice", [Pure]> {
  let summary = "Insert a strided slice into a tl.tensor (mixed static/dynamic offsets, static sizes/strides)";
  let arguments = (ins TT_Tensor:$src,
                       TT_Tensor:$tile,
                       Variadic<DsaSliceOffsetType>:$offsets,
                       DenseI64ArrayAttr:$static_offsets,
                       DenseI64ArrayAttr:$sizes,
                       DenseI64ArrayAttr:$strides);
  let results = (outs TT_Tensor:$result);
}

#endif // TLE_DSA_OPS
