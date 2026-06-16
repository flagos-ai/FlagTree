#ifndef TRITON_ILUVATARGPU_OPS
#define TRITON_ILUVATARGPU_OPS

include "mlir/IR/OpBase.td"
include "triton/Dialect/Triton/IR/TritonDialect.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUOpInterfaces.td"

include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "TritonILUVATARGPUDialect.td"


class TT_ILUVATARGPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<TritonILUVATARGPU_Dialect, mnemonic, !listconcat(traits, [])>;

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;
def SharedMemory : Resource<"::mlir::triton::gpu::SharedMemory">;

//===----------------------------------------------------------------------===//
// MaskedLoadOp
//===----------------------------------------------------------------------===//
def MaskedLoadOp : TT_ILUVATARGPU_Op<"masked_load", []> {
  let summary = "Masked load operation";
  let description = [{
    Load operation with masking and multicast support. If the mask is true, loads from the given pointer. Works with LLVM types as a utility op for making LLVM conversion easier.
    On architectures supporting multicast, the `multicastMask`specifies which CTAs in the cluster request the same data. This allows the hardware to efficiently broadcast the
    data to multiple CTAs in the cluster.
  }];
  let arguments = (ins
    LLVM_AnyPointer:$ptr,
    I1:$mask,
    LLVM_Type:$falseVal,
    Optional<I16>:$multicastMask,
    DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<BoolAttr, "false">:$forceNoAlias,
    DefaultValuedAttr<BoolAttr, "false">:$isVolatile
  );

  let results = (outs LLVM_Type:$result);

  let assemblyFormat = [{
    $ptr `,` $mask `,` $falseVal (`,` $multicastMask^)?
    oilist(`cacheModifier` `=` $cache)
    (`forceNoAlias` $forceNoAlias^)?
    (`isVolatile` $isVolatile^)?
    attr-dict `:` functional-type(operands, results)
  }];
}

//===----------------------------------------------------------------------===//
// MaskedStoreOp
//===----------------------------------------------------------------------===//
def MaskedStoreOp : TT_ILUVATARGPU_Op<"masked_store", []> {
  let summary = "Masked Store operation";
  let description = [{
    Store operation with masking support. If the mask is true, Store from the given pointer. Works with LLVM types as a utility op for making LLVM conversion easier.
  }];
  let arguments = (ins
    LLVM_AnyPointer:$ptr,
    LLVM_Type:$value,
    I1:$mask,
    DefaultValuedAttr<TT_CacheModifierAttr, "::mlir::triton::CacheModifier::NONE">:$cache,
    DefaultValuedAttr<BoolAttr, "false">:$forceNoAlias
  );

  let assemblyFormat = [{
    $ptr `,` $value `,` $mask
    oilist(`cacheModifier` `=` $cache)
    (`forceNoAlias` $forceNoAlias^)?
    attr-dict `:` type(operands)
  }];
}


#endif
