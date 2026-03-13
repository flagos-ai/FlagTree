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
  let arguments = (ins DsaRemotePointerType:$src, DsaRemoteShardIdType:$shard_id);
  let results = (outs DsaRemotePointerType:$result);
}

#endif // TLE_DSA_OPS
