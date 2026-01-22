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

class Tle_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tle_Dialect, mnemonic,
       !listconcat(traits, [TensorSizeTrait, VerifyTensorLayoutsTrait])> {
}

// Begin TLE-Lite

def Tle_ExtractTileOp : Tle_Op<"extract_tile", [Pure,
                                               SameOperandsAndResultElementType]> {
  let arguments = (ins TT_Tensor:$src, TT_IntLike:$index);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $src `[` $index `]` attr-dict `:` qualified(type($src)) `,` qualified(type($index)) `->` qualified(type($result))
  }];
  let builders = [OpBuilder<(ins "Value":$src, "Value":$index, "ArrayRef<int64_t>":$tile_shape)>];
}

def Tle_InsertTileOp : Tle_Op<"insert_tile", [Pure,
                                              DeclareOpInterfaceMethods<InferTypeOpInterface>,
                                              SameOperandsAndResultType]> {
  let arguments = (ins TT_Tensor:$src, TT_Tensor:$tile ,TT_IntLike:$index);
  let results = (outs TT_Tensor:$result);
  let assemblyFormat = [{
    $src `[` $index `]` `=` $tile attr-dict `:` qualified(type($index)) `,` qualified(type($tile)) `->` qualified(type($result))
  }];
}

// End TLE-Lite

// Begin TLE-Raw

def Tle_TensorType : AnyTypeOf<[TT_Type, TTG_MemDescType]>;
def Tle_ArgType : AnyTypeOf<[Tle_TensorType, LLVMPointerType, LLVMStructType]>;

def Tle_DSLRegionOp : Tle_Op<"dsl_region", [IsolatedFromAbove, MemDescViewTrait, RecursiveMemoryEffects]> {
  let arguments = (ins Variadic<Tle_ArgType>:$inputs,
    OptionalAttr<ArrayAttr>:$edsl_param_types,
    OptionalAttr<ArrayAttr>:$edsl_param_names);
  let results = (outs Variadic<Tle_ArgType>:$outputs);
  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;
}

def Tle_YieldOp : Tle_Op<"yield", [Pure, Terminator, ReturnLike, HasParent<"DSLRegionOp">]> {
  let arguments = (ins Variadic<Tle_ArgType>:$inputs);
  let assemblyFormat = "attr-dict ($inputs^ `:` type($inputs))?";
}

def Tle_ExtractAllocatedPtrOp : Tle_Op<"extract_allocated_ptr", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins Tle_ArgType:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_ExtractAlignedPtrOp : Tle_Op<"extract_aligned_ptr", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins Tle_ArgType:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_ExtractOffsetOp : Tle_Op<"extract_offset", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins Tle_TensorType:$input);
  let results = (outs I64:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_ExtractSizesOp : Tle_Op<"extract_sizes", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins Tle_TensorType:$input);
  let results = (outs Variadic<I64>:$sizes);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($sizes)";
  let builders = [OpBuilder<(ins "size_t":$num, "Value":$input)>];
}

def Tle_ExtractStridesOp : Tle_Op<"extract_strides", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins Tle_TensorType:$input);
  let results = (outs Variadic<I64>:$strides);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($strides)";
  let builders = [OpBuilder<(ins "size_t":$num, "Value":$input)>];
}

def Tle_ExtractPtrOp : Tle_Op<"extract_ptr", [Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins TT_Ptr:$input);
  let results = (outs LLVMPointerType:$ptr);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($ptr)";
}

def Tle_PackOp : Tle_Op<"pack", [MemDescViewTrait, Pure, HasParent<"DSLRegionOp">]> {
  let arguments = (ins LLVMStructType:$input);
  let results = (outs Tle_TensorType:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
  let hasVerifier = 1;
}

// End TLE-Raw

#endif
