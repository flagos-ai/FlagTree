#ifndef MTGPU_OPS
#define MTGPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MTGPUDialect.td"
include "MTGPUTypes.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class MTGPU_Op<string mnemonic, list<Trait> traits = []>
    : LLVM_OpBase<MTGPUDialect, mnemonic, traits>;

def MTGPU_SQMMA_LayoutAttr : I32EnumAttr<"SQMMALayout",
    "sqmma layout, either 'row' or 'col'",
    [
      I32EnumAttrCase<"row", 0>,
      I32EnumAttrCase<"col", 1>
    ]> {
  let cppNamespace = "::mlir::triton::mtgpu";
}

def MTGPU_SQMMA_EltTypeAttr : I32EnumAttr<"SQMMAEltType",
    "sqmma operand type, either 's8', 's32', 'e4m3', 'e5m2', 'f16', 'bf16', 'tf32', or 'f32'",
    [
      I32EnumAttrCase<"s8", 0>,
      I32EnumAttrCase<"s32", 1>,
      I32EnumAttrCase<"e4m3", 2>,
      I32EnumAttrCase<"e5m2", 3>,
      I32EnumAttrCase<"f16", 4>,
      I32EnumAttrCase<"bf16", 5>,
      I32EnumAttrCase<"tf32", 6>,
      I32EnumAttrCase<"f32", 7>
    ]> {
  let cppNamespace = "::mlir::triton::mtgpu";
}

def MTGPU_SQMMA_AccumulationModeAttr : I32EnumAttr<"SQMMAAccumulationMode",
    "sqmma accumulation contract",
    [
      I32EnumAttrCase<"hardware", 0>,
      I32EnumAttrCase<"partial", 1>,
      I32EnumAttrCase<"software", 2>
    ]> {
  let cppNamespace = "::mlir::triton::mtgpu";
}

def MTGPU_SQMMATensor : Type<
    And<[CPred<"::llvm::isa<::mlir::RankedTensorType>($_self)">,
         CPred<"::mlir::isa_and_nonnull<::mlir::triton::gpu::MUSASqmmaEncodingAttr>(::llvm::cast<::mlir::RankedTensorType>($_self).getEncoding())">]>,
    "MUSA SQMMA accumulator tensor">;
def MTGPU_SQMMACarrierType : Type<
    CPred<"::llvm::isa<::mlir::triton::mtgpu::SqmmaAccumulatorType>($_self)">,
    "MUSA SQMMA accumulator carrier">;
def MTGPU_SQMMACarrierOrTensorOrMemDesc : AnyTypeOf<
    [MTGPU_SQMMACarrierType, TTG_TensorOrMemDesc],
    "MUSA SQMMA carrier/tensor/memdesc wait value">;

def MTGPU_SqmmaOp : MTGPU_Op<"sqmma", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    TypesMatchWith<"result's type matches accumulator's type", "d", "c", "$_self">
  ]> {
  let arguments = (ins TTG_TensorOrMemDesc:$a, TTG_TensorOrMemDesc:$b,
                   MTGPU_SQMMACarrierType:$c, Optional<I1>:$useC,
                   I32Attr:$m, I32Attr:$n, I32Attr:$k,
                   MTGPU_SQMMA_EltTypeAttr:$eltTypeC,
                   MTGPU_SQMMA_EltTypeAttr:$eltTypeA,
                   MTGPU_SQMMA_EltTypeAttr:$eltTypeB,
                   MTGPU_SQMMA_LayoutAttr:$layoutA,
                   MTGPU_SQMMA_LayoutAttr:$layoutB,
                   DefaultValuedAttr<BoolAttr, "false">:$isAsync,
                   DefaultValuedAttr<MTGPU_SQMMA_AccumulationModeAttr,
                                     "::mlir::triton::mtgpu::SQMMAAccumulationMode::hardware">:$accMode,
                   DefaultValuedAttr<I32Attr, "0">:$inputPrecision,
                   DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc);
  let results = (outs MTGPU_SQMMACarrierType:$d);
  let assemblyFormat =
      "$a `,` $b `,` $c (`,` $useC^)? attr-dict `:` type($a) `*` type($b) `->` type($d)";
  let extraClassDeclaration = [{
    bool needsPartialAccumulator();
    bool usesSoftwareAccumulator();
    bool usesHardwareAccumulator();
  }];
  let hasVerifier = 1;
}

def MTGPU_PackSqmmaAccumulatorOp : MTGPU_Op<"pack_sqmma_accumulator", [Pure]> {
  let summary = "Pack a SQMMA accumulator tensor into compact carrier form";
  let arguments = (ins MTGPU_SQMMATensor:$input);
  let results = (outs MTGPU_SQMMACarrierType:$carrier);
  let assemblyFormat =
      "$input attr-dict `:` type($input) `->` type($carrier)";
  let hasVerifier = 1;
}

def MTGPU_UnpackSqmmaAccumulatorOp : MTGPU_Op<"unpack_sqmma_accumulator", [Pure]> {
  let summary = "Unpack a SQMMA accumulator carrier back to tensor form";
  let arguments = (ins MTGPU_SQMMACarrierType:$carrier);
  let results = (outs MTGPU_SQMMATensor:$output);
  let assemblyFormat =
      "$carrier attr-dict `:` type($carrier) `->` type($output)";
  let hasVerifier = 1;
}

def MTGPU_SqmmaWaitOp : MTGPU_Op<"sqmma_wait", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    AllTypesMatch<["inputs", "outputs"]>,
    PassthroughWaitLike,
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "Wait until pending SQMMA operations complete.";
  let arguments = (ins Variadic<MTGPU_SQMMACarrierOrTensorOrMemDesc>:$inputs);
  let results = (outs Variadic<MTGPU_SQMMACarrierOrTensorOrMemDesc>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

#endif
