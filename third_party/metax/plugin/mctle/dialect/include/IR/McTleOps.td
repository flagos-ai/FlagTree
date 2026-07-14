#ifndef MCTLE_OPS
#define MCTLE_OPS

include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mctle/dialect/include/IR/McTleDialect.td"
include "mctle/dialect/include/IR/McTleAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class McTle_Op<string mnemonic, list<Trait> traits = []>
    : Op<McTle_Dialect, mnemonic,
         !listconcat(traits, [VerifyTensorLayoutsTrait])> {}

// Begin TLE-Lite
def McTle_ExtractTileOp : McTle_Op<"extract_tile", [Pure]> {
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

def McTle_InsertTileOp
    : McTle_Op<"insert_tile", [Pure,
                               DeclareOpInterfaceMethods<InferTypeOpInterface>,
]> {
  let arguments = (ins TT_Tensor:$src, TT_Tensor:$tile, TT_IntLike:$index);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $src `[` $index `]` `=` $tile attr-dict `:` qualified(type($src)) `,` qualified(type($index)) `,` qualified(type($tile)) `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

// End TLE-Lite

// Begin TLE-Raw

def McTle_TensorType : AnyTypeOf<[TT_Type, TTG_MemDescType]>;
def McTle_ArgType : AnyTypeOf<[McTle_TensorType, LLVMPointerType, LLVMStructType]>;
def McTle_LocalPointerResultType : AnyTypeOf<[TT_Tensor, TT_Ptr]>;
def McTle_LocalPointerIndexType : AnyTypeOf<[TT_Tensor, TT_Int]>;

def McTle_LocalPointersOp : McTle_Op<"local_pointers", [Pure]> {
  let arguments = (ins TTG_MemDescType:$src,
      Variadic<McTle_LocalPointerIndexType>:$indices);
  let results = (outs McTle_LocalPointerResultType:$result);
  let hasVerifier = 1;
}

#endif
