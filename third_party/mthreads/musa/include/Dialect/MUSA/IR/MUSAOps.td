#ifndef MUSA_OPS
#define MUSA_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "MUSADialect.td"
include "MUSAAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonOpInterfaces.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUAttrBase.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypeInterfaces.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td"

class MUSA_Op<string mnemonic, list<Trait> traits = []> :
    LLVM_OpBase<MUSA_Dialect, mnemonic, traits>;

def LLVM_PointerShared : LLVM_PointerInAddressSpace<3>;

def SQMMA_LayoutAttr : I32EnumAttr<"SQMMALayout",
    "sqmma layout, either 'row' or 'col'",
    [
      I32EnumAttrCase<"row", 0>,
      I32EnumAttrCase<"col", 1>
    ]> {
  let cppNamespace = "::mlir::triton::musa";
}

def SQMMA_EltTypeAttr : I32EnumAttr<"SQMMAEltType",
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
  let cppNamespace = "::mlir::triton::musa";
}

def SQMMA_AccumulationModeAttr : I32EnumAttr<"SQMMAAccumulationMode",
    "sqmma accumulation contract",
    [
      I32EnumAttrCase<"hardware", 0>,
      I32EnumAttrCase<"partial", 1>,
      I32EnumAttrCase<"software", 2>
    ]> {
  let cppNamespace = "::mlir::triton::musa";
}

def TME_DimType : AnyTypeOf<[I32, LLVM_AnyVector], "TME block dim/pos type">;
def TME_DescType : AnyTypeOf<[I64, TT_TensorDescType],
    "TME descriptor handle or tensor descriptor">;
def TME_LocalMemType : AnyTypeOf<[LLVM_PointerShared, TTG_MemDescType],
    "shared memory pointer or memdesc for TME async copy">;

def MUSA_SquadDotOp : MUSA_Op<"squad_dot", [
    DeclareOpInterfaceMethods<DotOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    TypesMatchWith<"result's type matches accumulator's type",
                   "d", "c", "$_self">
  ]> {
  let arguments = (ins TTG_TensorOrMemDesc:$a, TTG_TensorOrMemDesc:$b,
                   TT_FpIntTensor:$c, Optional<I1>:$useC,
                   I32Attr:$m, I32Attr:$n, I32Attr:$k,
                   SQMMA_EltTypeAttr:$eltTypeC, SQMMA_EltTypeAttr:$eltTypeA,
                   SQMMA_EltTypeAttr:$eltTypeB,
                   SQMMA_LayoutAttr:$layoutA, SQMMA_LayoutAttr:$layoutB,
                   DefaultValuedAttr<BoolAttr, "false">:$isAsync,
                   DefaultValuedAttr<SQMMA_AccumulationModeAttr,
                                     "::mlir::triton::musa::SQMMAAccumulationMode::hardware">:$accMode,
                   DefaultValuedAttr<I32Attr, "0">:$inputPrecision,
                   DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc);
  let results = (outs TT_FpIntTensor:$d);
  let assemblyFormat =
      "$a `,` $b `,` $c (`,` $useC^)? attr-dict `:` type($a) `*` type($b) `->` type($d)";
  let extraClassDeclaration = [{
    bool needsPartialAccumulator();
    bool usesSoftwareAccumulator();
    bool usesHardwareAccumulator();
  }];
  let hasVerifier = 1;
}

def MUSA_SquadDotWaitOp : MUSA_Op<"squad_dot_wait", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    AllTypesMatch<["inputs", "outputs"]>,
    PassthroughWaitLike,
    MemoryEffects<[MemRead, MemWrite]>
  ]> {
  let summary = "Wait until pending SQMMA operations complete.";
  let arguments = (ins Variadic<TTG_TensorOrMemDesc>:$inputs);
  let results = (outs Variadic<TTG_TensorOrMemDesc>:$outputs);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs)";
  let hasCanonicalizeMethod = 1;
  let hasVerifier = 1;
}

def MUSA_WmmaDotOp : MUSA_Op<"wmma_dot", [
    DeclareOpInterfaceMethods<DotOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    TypesMatchWith<"result's type matches accumulator's type",
                   "d", "c", "$_self">
  ]> {
  let arguments = (ins TT_FpIntTensor:$a, TT_FpIntTensor:$b,
                   TT_FpIntTensor:$c, Optional<I1>:$useC,
                   I32Attr:$m, I32Attr:$n, I32Attr:$k,
                   SQMMA_EltTypeAttr:$eltTypeA, SQMMA_EltTypeAttr:$eltTypeB,
                   SQMMA_LayoutAttr:$layoutA, SQMMA_LayoutAttr:$layoutB,
                   DefaultValuedAttr<I32Attr, "0">:$inputPrecision,
                   DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc);
  let results = (outs TT_FpIntTensor:$d);
  let assemblyFormat =
      "$a `,` $b `,` $c (`,` $useC^)? attr-dict `:` type($a) `*` type($b) `->` type($d)";
  let extraClassDeclaration = [{
    bool needsPartialAccumulator();
  }];
  let hasVerifier = 1;
}

def MUSA_WmmaDotWaitOp : MUSA_Op<"wmma_dot_wait", []> {
  let summary = "IR no-op marker for WMMA dependency edges.";
  let assemblyFormat = "attr-dict";
}

def MUSA_BarRecordOp : MUSA_Op<"bar_record", []> {
  let summary = "Record an async barrier id before use.";
  let arguments = (ins I32:$barId);
  let assemblyFormat = "$barId attr-dict `:` type($barId)";
  let hasVerifier = 1;
}

def MUSA_InitArrivalOp : MUSA_Op<"init_arrival", []> {
  let summary = "Initialize async barrier arrival count and phase id.";
  let arguments = (ins I32:$barId, I32:$arriveCount, I32:$phaseId);
  let assemblyFormat =
      "$barId `,` $arriveCount `,` $phaseId attr-dict `:` type($barId)";
  let hasVerifier = 1;
}

def MUSA_BarrierAddTransOp : MUSA_Op<"barrier_add_trans", []> {
  let summary = "Explicitly add async barrier transaction bytes.";
  let arguments = (ins I32:$barId, I32:$transBytes, I1:$pred);
  let assemblyFormat =
      "$barId `,` $transBytes `,` $pred attr-dict `:` type($barId)";
  let hasVerifier = 1;
}

def MUSA_ArriveBarrierOp : MUSA_Op<"arrive_barrier", []> {
  let summary = "Arrive on async barrier and return phase id.";
  let arguments = (ins I32:$barId);
  let results = (outs I32:$phaseId);
  let assemblyFormat = "$barId attr-dict `:` type($barId) `->` type($phaseId)";
  let hasVerifier = 1;
}

def MUSA_ArriveBarrierNoRetOp : MUSA_Op<"arrive_barrier_noret", []> {
  let summary = "Arrive on async barrier without returning phase id.";
  let arguments = (ins I32:$barId, I1:$pred);
  let assemblyFormat = "$barId `,` $pred attr-dict `:` type($barId)";
  let hasVerifier = 1;
}

def MUSA_WaitBarrierOp : MUSA_Op<"wait_barrier", [
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
  ]> {
  let summary = "Wait on async barrier phase.";
  let arguments = (ins I32:$barId, I32:$phaseId);
  let assemblyFormat =
      "$barId `,` $phaseId attr-dict `:` type($barId)";
  let hasVerifier = 1;
}

def MUSA_AsyncTMECopyGlobalToLocalOp
    : MUSA_Op<"async_tme_copy_global_to_local", [
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Asynchronously copy a TME tile from global to shared.";
  let arguments = (ins
    TME_DescType:$desc,
    Variadic<I32>:$coord,
    I32:$barId,
    TTG_MemDescType:$result,
    I1:$pred,
    DenseI32ArrayAttr:$blockShape,
    TME_SwizzleGranularityAttr:$swizzleGranularity,
    TME_SwizzleStrideAttr:$swizzleStride,
    TME_SwizzleLineAttr:$swizzleLine,
    TME_PrefetchSizeAttr:$prefetchSize,
    TME_CachePolicyAttr:$cachePolicy,
    TME_PersistenceAttr:$innerPersistence,
    TME_PersistenceAttr:$outerPersistence
  );
  let assemblyFormat = [{
    $desc `[` $coord `]` `,` $barId `,` $result `,` $pred attr-dict
    `:` qualified(type($desc)) `,` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def MUSA_AsyncTMECopyLocalToGlobalOp
    : MUSA_Op<"async_tme_copy_local_to_global", [
        DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
      ]> {
  let summary = "Asynchronously copy a TME tile from shared to global.";
  let arguments = (ins
    TME_DescType:$desc,
    Variadic<I32>:$coord,
    TTG_MemDescType:$src,
    I1:$pred,
    DenseI32ArrayAttr:$blockShape,
    TME_SwizzleGranularityAttr:$swizzleGranularity,
    TME_SwizzleStrideAttr:$swizzleStride,
    TME_SwizzleLineAttr:$swizzleLine,
    TME_CachePolicyAttr:$cachePolicy,
    TME_PersistenceAttr:$innerPersistence,
    TME_PersistenceAttr:$outerPersistence
  );
  let assemblyFormat = [{
    $desc `[` $coord `]` `,` $src `,` $pred attr-dict
    `:` qualified(type($desc)) `,` qualified(type($src))
  }];
  let hasVerifier = 1;
}

def MUSA_TMEStoreCommitOp : MUSA_Op<"tme_store_commit", []> {
  let summary = "Commit pending TME stores.";
  let assemblyFormat = "attr-dict";
}

def MUSA_TMEStoreReadWaitOp : MUSA_Op<"tme_store_read_wait", []> {
  let summary = "Wait until committed TME stores are visible to reads.";
  let assemblyFormat = "attr-dict";
}

#endif
