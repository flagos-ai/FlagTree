#ifndef FLAGTREE_OPS
#define FLAGTREE_OPS

include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "triton/Dialect/FlagTree/IR/FlagTreeDialect.td"
include "triton/Dialect/FlagTree/IR/FlagTreeAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class FlagTree_Op<string mnemonic, list<Trait> traits = []> :
    Op<FlagTree_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def FlagTree_TensorType : AnyTypeOf<[TT_Type, TTG_MemDescType]>;
def FlagTree_ArgType : AnyTypeOf<[FlagTree_TensorType, LLVMPointerType, LLVMStructType]>;

def FlagTree_DSLRegionOp : FlagTree_Op<"dsl_region", [IsolatedFromAbove, MemDescViewTrait, RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<FlagTree_ArgType>:$inputs);
  let results = (outs Variadic<FlagTree_ArgType>:$outputs);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def FlagTree_YieldOp : FlagTree_Op<"yield", [Pure, Terminator, ReturnLike, HasParent<"DSLRegionOp">]> {
  let arguments = (ins Variadic<FlagTree_ArgType>:$inputs);
  let assemblyFormat = "attr-dict ($inputs^ `:` type($inputs))?";
}

def FlagTree_ExtractAllocatedPtrOp : FlagTree_Op<"extract_allocated_ptr", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins FlagTree_ArgType:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def FlagTree_ExtractAlignedPtrOp : FlagTree_Op<"extract_aligned_ptr", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins FlagTree_ArgType:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def FlagTree_ExtractOffsetOp : FlagTree_Op<"extract_offset", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins FlagTree_TensorType:$input);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def FlagTree_ExtractSizesOp : FlagTree_Op<"extract_sizes", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins FlagTree_TensorType:$input);
  let results = (outs Variadic<I64>:$sizes);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($sizes)";
  let builders = [OpBuilder<(ins "size_t":$num, "Value":$input)>];
}

def FlagTree_ExtractStridesOp : FlagTree_Op<"extract_strides", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins FlagTree_TensorType:$input);
  let results = (outs Variadic<I64>:$strides);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($strides)";
  let builders = [OpBuilder<(ins "size_t":$num, "Value":$input)>];
}

def FlagTree_ExtractPtrOp : FlagTree_Op<"extract_ptr", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins TT_Ptr:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def FlagTree_PackOp : FlagTree_Op<"pack", [MemDescViewTrait, Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins LLVMStructType:$input);
  let results = (outs FlagTree_TensorType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let hasVerifier = 1;
}

#endif
