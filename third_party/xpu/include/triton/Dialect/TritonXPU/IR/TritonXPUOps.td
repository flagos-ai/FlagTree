#ifndef TRITONXPU_OPS
#define TRITONXPU_OPS

include "mlir/IR/OpBase.td"                  // Trait
include "mlir/Dialect/Arith/IR/ArithBase.td" // Arith_CmpFPredicateAttr
include
    "triton/Dialect/TritonXPU/IR/TritonXPUDialect.td" // For TritonXPU_Dialect
include "triton/Dialect/TritonXPU/IR/TritonXPUAttrDefs.td"   // For Attr
include "triton/Dialect/TritonXPU/IR/TritonXPUEnums.td"      // For XPU op enums
include "triton/Dialect/TritonXPU/IR/TritonXPUTypes.td"      // For types
include "triton/Dialect/TritonXPU/IR/TritonXPUInterfaces.td" // For types
include "triton/Dialect/TritonGPU/IR/TritonGPUTypes.td" // For TTG_MemDescType (TLE ops)

//===----------------------------------------------------------------------===//
// TRITONXPU op definitions
//===----------------------------------------------------------------------===//

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;

class TTX_Op<string mnemonic, list<Trait> traits = []>
    : Op<TritonXPU_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Memory Ops
//===----------------------------------------------------------------------===//

def TTX_AllocaOp : TTX_Op<"alloca"> {
  let summary = "alloca";
  let arguments = (ins SI64Attr:$size);
  let results = (outs TT_PtrLike:$result);
}

def TTX_GM2LMOp : TTX_Op<"gm2lm", [AttrSizedOperandSegments,
                                   MemoryEffects<[MemRead<GlobalMemory>]>]> {
  let summary = "gm2lm";
  // [sync source bc5e33a6] LayerNorm vectorization can turn `other` into a
  // tensor<vector<T>>. The sync source widened shared TT_Type, but FlagTree keeps
  // main-tree TableGen definitions pristine. TTX_Type is the XPU-local
  // equivalent and already admits vectors/tensors of vectors.
  let arguments = (ins TT_PtrLike:$ptr, Optional<TT_IntLike>:$len,
      Optional<TTX_Type>:$other, Optional<TT_PtrLike>:$bufPtr,
      SI32Attr:$offsetState, SI32Attr:$fixedStride, SI64Attr:$rowLen,
      SI64Attr:$rowStride, SI32Attr:$lrie, SI32Attr:$tensorColSize,
      BoolAttr:$SVOpt, TT_SyncAttr:$syncMode, BoolAttr:$atomicSim,
      BoolAttr:$cache, BoolAttr:$handwrittenOffsetState);
  let results = (outs TT_PtrLike:$result);
}

def TTX_LM2GMOp : TTX_Op<"lm2gm", [AttrSizedOperandSegments,
                                   MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "lm2gm";
  let arguments = (ins TT_PtrLike:$ptr, TTX_UIVectorType:$value,
      Optional<TT_IntLike>:$len, Optional<TT_PtrLike>:$bufPtr,
      SI32Attr:$tensorColSize, SI32Attr:$offsetState, SI64Attr:$rowLen,
      SI64Attr:$rowStride, TT_SyncAttr:$syncMode, BoolAttr:$atomicSim,
      BoolAttr:$handwrittenOffsetState);
}

def TTX_SM2GMOp : TTX_Op<"sm2gm", [AttrSizedOperandSegments]> {
  let summary = "sm2gm";
  let arguments = (ins TT_PtrLike:$ptr, Optional<TTX_Type>:$value,
      Optional<TT_IntLike>:$len, Optional<TT_PtrLike>:$bufPtr,
      SI32Attr:$offsetState);
}

def TTX_GM2LMMaskOp
    : TTX_Op<"gm2lm_mask", [AttrSizedOperandSegments,
                            MemoryEffects<[MemRead<GlobalMemory>]>]> {
  let summary = "gm2lm";
  let arguments = (ins TT_PtrLike:$ptr, Optional<TT_BoolLike>:$mask,
      // Keep the masked form consistent with gm2lm above.
      Optional<TTX_Type>:$other, Optional<TT_IntLike>:$len,
      Optional<TT_PtrLike>:$bufPtr, SI32Attr:$offsetState,
      SI32Attr:$fixedStride, SI64Attr:$rowLen, SI64Attr:$rowStride,
      SI32Attr:$lrie, SI32Attr:$tensorColSize, BoolAttr:$SVOpt,
      TT_SyncAttr:$syncMode, BoolAttr:$atomicSim, BoolAttr:$cache,
      BoolAttr:$handwrittenOffsetState);
  let results = (outs TT_PtrLike:$result);
}

def TTX_LM2GMMaskOp
    : TTX_Op<"lm2gm_mask", [AttrSizedOperandSegments,
                            MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "lm2gm";
  let arguments = (ins TT_PtrLike:$ptr, TTX_UIVectorType:$value,
      Optional<TT_BoolLike>:$mask, Optional<TT_IntLike>:$len,
      Optional<TT_PtrLike>:$bufPtr, SI32Attr:$tensorColSize,
      SI32Attr:$offsetState, SI64Attr:$rowLen, SI64Attr:$rowStride,
      TT_SyncAttr:$syncMode, BoolAttr:$atomicSim,
      BoolAttr:$handwrittenOffsetState);
}

def TTX_SM2GMMaskOp : TTX_Op<"sm2gm_mask", [AttrSizedOperandSegments]> {
  let summary = "sm2gm";
  let arguments = (ins TT_PtrLike:$ptr, Optional<TTX_Type>:$value,
      Optional<TT_BoolLike>:$mask, Optional<TT_IntLike>:$len,
      Optional<TT_PtrLike>:$bufPtr, SI32Attr:$offsetState);
}

def TTX_LoadOp
    : TTX_Op<"load", [AttrSizedOperandSegments, MemoryEffects<[MemRead]>]> {
  let summary = "load";
  let arguments = (ins TT_PtrLike:$ptr, Optional<TT_BoolLike>:$mask,
      // [sync source bc5e33a6] Same as TTX_GM2LMOp.$other: vectorize
      // can widen `other` to tensor<vector<T>>. The sync source relied on
      // its widened shared TT_Type; FlagTree keeps main-tree TT_Type
      // pristine and uses the XPU-local TTX_Type (= TT_Type | vector).
      Optional<TTX_Type>:$other, Optional<TT_Int>:$index, SI32Attr:$stride,
      SI32Attr:$tensorColSize, BoolAttr:$isDiscrete, BoolAttr:$SVOpt,
      BoolAttr:$bf16Tofp32Unordered, TT_SyncAttr:$syncMode);
  let results = (outs TTX_Type:$result);
}

def TTX_StageSMOp
    : TTX_Op<"stage_sm", [MemoryEffects<[MemRead<GlobalMemory>, MemWrite]>]> {
  let summary = "stage a loop-invariant GM array into cluster shared memory "
                "(SM) once, shared by all cores";
  let description = [{
      Copies a loop-invariant, read-only GM array into a cluster-shared SM
      buffer once, outside the grid loop. Only core 0 issues the GM->SM DMA;
      all cores then synchronize on a cluster barrier before reading. The
      result is a scalar SM base pointer (addrspace 2) that subsequent
      `triton_xpu.load_scalar_indexed` ops read from.
    }];
  let arguments = (ins TT_Ptr:$ptr, Optional<I32>:$len, I32:$smOffset,
      I32:$bufElems, TT_SyncAttr:$syncMode);
  let results = (outs TT_Ptr:$result);
}

def TTX_LoadScalarIndexedOp
    : TTX_Op<"load_scalar_indexed", [MemoryEffects<[MemRead]>]> {
  let summary = "load one scalar from a staged buffer at a runtime element "
                "index and broadcast it to the result tensor";
  let arguments = (ins TT_PtrLike:$ptr, TT_Int:$index, TT_SyncAttr:$syncMode);
  let results = (outs TT_Type:$result);
}

def TTX_StoreOp
    : TTX_Op<"store", [AttrSizedOperandSegments, MemoryEffects<[MemWrite]>]> {
  let summary = "store";
  let arguments = (ins TT_PtrLike:$ptr, TTX_UIVectorType:$value,
      Optional<TT_BoolLike>:$mask, Optional<TT_Int>:$index,
      SI32Attr:$tensorColSize, BoolAttr:$bf16Tofp32Unordered,
      TT_DtypeAttr:$dtype, TT_SyncAttr:$syncMode);
}

def TTX_StoreSMOp
    : TTX_Op<"storeSM", [MemoryEffects<[MemWrite]>, AttrSizedOperandSegments]> {
  let summary = "store to SM";
  // [sync source bc5e33a6] `value` may be a vectorized tensor<vector<T>>; use the
  // XPU-local TTX_Type (= TT_Type | vector) instead of pristine main-tree
  // TT_Type.
  let arguments = (ins TT_PtrLike:$ptr, TTX_Type:$value,
      Optional<TT_BoolLike>:$mask, Optional<TT_Int>:$index);
}

//===----------------------------------------------------------------------===//
// Context Ops
//===----------------------------------------------------------------------===//

def TTX_GetCoreIdOp : TTX_Op<"get_core_id"> {
  let summary = "get_core_id";
  let results = (outs TTX_Type:$result);
}

def TTX_GetThreadIdOp : TTX_Op<"get_thread_id"> {
  let summary = "get_thread_id";
  let description = [{
        threadType(0): tid = core_id() * cluster_num() + cluster_id()
        threadType(1): tid = core_num() * cluster_id() + core_id()
    }];
  let arguments = (ins SI32Attr:$threadType);
  let results = (outs TTX_Type:$result);
}

def TTX_GetClusterIdOp : TTX_Op<"get_cluster_id", [NoMemoryEffect]> {
  let summary = "Get the ID of the current Cluster";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def TTX_GetNumClusterOp : TTX_Op<"get_num_cluster", [NoMemoryEffect]> {
  let summary = "Get the number of launched Cluster";
  let results = (outs I32:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// MakeRange-Liks Ops
//===----------------------------------------------------------------------===//
// TODO[dyq]: combine TTX_MakeRangeOp && TTX_InterleaveOp && TTX_OutRangeOp

// borrowed from triton/include/triton/Dialect/Triton/IR/TritonOps.td
def TTX_MakeRangeOp : TTX_Op<"make_range", [AttrSizedOperandSegments, Pure]> {
  let summary = "make range";

  let description = [{
        Returns an 1D int32 tensor.

        Values span from $start to $end (exclusive), with step = 1
    }];

  // WARNING: MLIR generates getStart()/getEnd() functions which return
  // uint32_t, even though these arguments are to be interpreted as *signed*
  // int32 values.  If this matters, use get{Start,End}Attr().getInt(), which
  // return int64_t.
  let arguments = (ins I32Attr:$start, I32Attr:$end, I32Attr:$realSize,
      Optional<TT_Int>:$loopIndex, Optional<TT_Int>:$unrollIndex);

  let results = (outs TT_IntTensor:$result);

  let assemblyFormat = "$loopIndex $unrollIndex attr-dict `:` type($loopIndex) "
                       "type($unrollIndex) `->` type($result)";

  // let hasFolder = 1;
  let hasVerifier = 1;
}

def TTX_InterleaveOp : TTX_Op<"interleave", [AttrSizedOperandSegments, Pure]> {
  let summary = "interleave";

  let description = [{
        Returns an 1D int32 tensor.

        Values span from $start to $end (exclusive), with step = 1
    }];

  // WARNING: MLIR generates getStart()/getEnd() functions which return
  // uint32_t, even though these arguments are to be interpreted as *signed*
  // int32 values.  If this matters, use get{Start,End}Attr().getInt(), which
  // return int64_t.
  let arguments = (ins I32Attr:$start, I32Attr:$end,
      Optional<TT_Int>:$loopIndex, Optional<TT_Int>:$unrollIndex);

  let results = (outs TT_IntTensor:$result);

  let assemblyFormat = "$loopIndex $unrollIndex attr-dict `:` type($loopIndex) "
                       "type($unrollIndex) `->` type($result)";

  // let hasFolder = 1;
  let hasVerifier = 1;
}

def TTX_OutRangeOp : TTX_Op<"out_range"> {
  let summary = "out_range";
  let description = [{
        Returns idx.
    }];
  let arguments = (ins I32Attr:$groupsize, I32Attr:$rowspercore, TT_Int:$index);
  let results = (outs TT_IntTensor:$result);
}

//===----------------------------------------------------------------------===//
// Vectorization Ops
//===----------------------------------------------------------------------===//

class TTX_VUnaryFOp<string mnemonic, list<Trait> traits = []>
    : TTX_Op<mnemonic, traits#[SameOperandsAndResultType]>,
      Arguments<(ins TTX_VectorLike:$value)>,
      Results<(outs TTX_VectorLike:$result)> {
  let assemblyFormat = "$value `,` attr-dict `:` type($result)";
}

class TTX_BinaryOp<string mnemonic, list<Trait> traits = []>
    : TTX_Op<mnemonic, traits#[SameOperandsAndResultType]> {
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class TTX_VVIntBinaryOp<string mnemonic, list<Trait> traits = []>
    : TTX_BinaryOp<mnemonic, traits>,
      Arguments<(ins TTX_VectorLike:$lhs, TTX_VectorLike:$rhs)>,
      Results<(outs TTX_VectorLike:$result)>;

class TTX_VVFloatBinaryOp<string mnemonic, list<Trait> traits = []>
    : TTX_BinaryOp<mnemonic, traits>,
      Arguments<(ins TTX_VectorLike:$lhs, TTX_VectorLike:$rhs)>,
      Results<(outs TTX_VectorLike:$result)>;

class TTX_SVFloatBinaryOp<string mnemonic, list<Trait> traits = []>
    : TTX_Op<mnemonic, traits>,
      Arguments<(ins TTX_Type:$lhs, TTX_Type:$rhs, I32Attr:$elemState)>,
      Results<(outs TTX_VectorLike:$result)> {
  let summary = mnemonic;
}

def TTX_VExpFOp : TTX_VUnaryFOp<"vexpf">;
def TTX_VLogFOp : TTX_VUnaryFOp<"vlogf">;
def TTX_VSinFOp : TTX_VUnaryFOp<"vsinf">;
def TTX_VCosFOp : TTX_VUnaryFOp<"vcosf">;
def TTX_VSqrtFOp : TTX_VUnaryFOp<"vsqrtf">;
def TTX_VAbsFOp : TTX_VUnaryFOp<"vabsf">;
def TTX_VSIToFPOp : TTX_Op<"vsitofp"> {
  let summary = "vsitofp";
  let arguments = (ins TTX_VectorLike:$value);
  let results = (outs TTX_VectorLike:$result);
}

def TTX_VvaddFOp : TTX_VVFloatBinaryOp<"vvaddf">;
def TTX_VvsubFOp : TTX_VVFloatBinaryOp<"vvsubf">;
def TTX_VvmulFOp : TTX_VVFloatBinaryOp<"vvmulf">;
def TTX_VvdivFOp : TTX_VVFloatBinaryOp<"vvdivf">;
def TTX_VvmaxFOp : TTX_VVFloatBinaryOp<"vvmaxf">;
def TTX_VvminFOp : TTX_VVFloatBinaryOp<"vvminf">;
def TTX_VvmaxNumFOp : TTX_VVFloatBinaryOp<"vvmaxnumf">;
def TTX_VvminNumFOp : TTX_VVFloatBinaryOp<"vvminnumf">;

def TTX_SvaddFOp : TTX_SVFloatBinaryOp<"svaddf">;
def TTX_SvmulFOp : TTX_SVFloatBinaryOp<"svmulf">;
def TTX_SvsubFOp : TTX_SVFloatBinaryOp<"svsubf">;
def TTX_SvmaxFOp : TTX_SVFloatBinaryOp<"svmaxf">;
def TTX_SvxorIOp : TTX_SVFloatBinaryOp<"svxori">;

def TTX_VvorIOp : TTX_VVIntBinaryOp<"vvori">;
def TTX_VvxorIOp : TTX_VVIntBinaryOp<"vvxori">;
def TTX_VvandIOp : TTX_VVIntBinaryOp<"vvandi">;
def TTX_VvaddIOp : TTX_VVIntBinaryOp<"vvaddi">;
def TTX_VvsubIOp : TTX_VVIntBinaryOp<"vvsubi">;
def TTX_VvmulIOp : TTX_VVIntBinaryOp<"vvmuli">;
// The signed/unsigned min-max and division forms exist for the combine
// regions UnrollControl inlines (tensor-of-vector operands): an
// arith.maxsi-style combine has no other vector form to be rewritten into.
def TTX_VvmaxSIOp : TTX_VVIntBinaryOp<"vvmaxsi">;
def TTX_VvminSIOp : TTX_VVIntBinaryOp<"vvminsi">;
def TTX_VvmaxUIOp : TTX_VVIntBinaryOp<"vvmaxui">;
def TTX_VvminUIOp : TTX_VVIntBinaryOp<"vvminui">;
def TTX_VvdivSIOp : TTX_VVIntBinaryOp<"vvdivsi">;
def TTX_VvdivUIOp : TTX_VVIntBinaryOp<"vvdivui">;

def TTX_VMacFOp : TTX_Op<"vmacf", [SameOperandsAndResultType]> {
  let summary = "vmacf";
  let arguments = (ins TTX_VectorLike:$value, TTX_VectorLike:$mulData,
      TTX_VectorLike:$addData);
  let results = (outs TTX_VectorLike:$result);
  let assemblyFormat =
      "$value `,` $mulData `,` $addData attr-dict `:` type($result)";
}

def TTX_VConstOp : TTX_Op<"vconst"> {
  let summary = "vconst";
  let arguments = (ins AnyAttr:$value);
  let results = (outs TTX_VectorLike:$result);
}

def TTX_VSplatOp : TTX_Op<"vsplat"> {
  let summary = "vsplat";
  let arguments = (ins TTX_Type:$src);
  let results = (outs TTX_VectorLike:$result);
}

def TTX_VSelectOp : TTX_Op<"vselect"> {
  let summary = "vselect";
  let arguments = (ins TTX_Type:$condition, TTX_Type:$true_value,
      TTX_Type:$false_value);
  let results = (outs TTX_VectorLike:$result);
}

def TTX_VExtFOp : TTX_Op<"vextf"> {
  let summary = "vextf";
  let arguments = (ins TTX_VectorLike:$value);
  let results = (outs TTX_VectorLike:$result);
}

def TTX_VTruncFOp : TTX_Op<"vtruncf"> {
  let summary = "vtruncf";
  let arguments = (ins TTX_VectorLike:$value);
  let results = (outs TTX_VectorLike:$result);
}

def TTX_VCmpFOp : TTX_Op<"vcmpf"> {
  let summary = "vcmpf";
  let arguments = (ins Arith_CmpFPredicateAttr:$predicate, TTX_VectorLike:$lhs,
      TTX_VectorLike:$rhs);
  let results = (outs TTX_UIVectorLike:$result);
}

// Integer counterpart of vcmpf. Without it, UnrollControl's inlined
// select(cmpi(...)) combine keeps an arith.cmpi on tensor-of-vector
// operands, which the verifier rejects (operand must be integer-like).
def TTX_VCmpIOp : TTX_Op<"vcmpi", [Pure]> {
    let summary = "vcmpi";
    let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                        TTX_VectorLike:$lhs,
                        TTX_VectorLike:$rhs);
    let results = (outs TTX_UIVectorLike:$result);
}

def TTX_CmpFOp : TTX_Op<"cmpf", [Pure]> {
    let summary = "cmpf";
    // The result is TTX_UIVectorType, not TTX_Type: the compare fusion in
    // tritonxpu-normalize builds this op with a ui32 result (the shape
    // TTX_StoreOp's $value already accepts) and tritonxpu-vectorize later
    // retypes it into a vcmpf, whose result constraint allows ui32 too. While
    // both halves lived in one pass the verifier never saw the intermediate;
    // splitting them (step 1.5) exposed it. Widening only what the verifier
    // accepts cannot change emitted code.
    let arguments = (ins Arith_CmpFPredicateAttr:$predicate,
                     TTX_Type:$lhs,
                     TTX_Type:$rhs);
    let results = (outs TTX_UIVectorType:$res);
}

//===----------------------------------------------------------------------===//
// Other Ops
//===----------------------------------------------------------------------===//

def TTX_MinOp : TTX_Op<"min"> {
  let summary = "min";
  let arguments = (ins TTX_Type:$lhs, TTX_Type:$rhs);
  let results = (outs TTX_Type:$res);
}

def TTX_ExtractOp : TTX_Op<"extract"> {
  let summary = "extract the element from the tensor";
  let description = [{
        Returns an 1D int32.
        Extract the element from the tensor according to index.
    }];
  let arguments = (ins I32Attr:$index, TTX_Type:$tensor);
  let results = (outs TTX_Type:$result);
}

def TTX_ExtractSliceOp : TTX_Op<"extract_slice"> {
  let summary = "extract slice a part of elements from the tensor";
  let description = [{
        Returns an 1D int32.
        Truncate a part of elements from the tensor.
    }];
  let arguments = (ins TTX_Type:$tensor);
  let results = (outs TTX_Type:$result);
}

// borrowed from triton/include/triton/Dialect/Triton/IR/TritonOps.td
def TTX_ConvertLayoutOp
    : TTX_Op<"convert_layout", [SameOperandsAndResultShape,
                                SameOperandsAndResultElementType, Pure]> {
  let summary = "convert layout";

  let arguments = (ins TTX_TensorVector:$src);

  let results = (outs TTX_TensorVector:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

// borrowed from triton/include/triton/Dialect/Triton/IR/TritonOps.td
def TTX_ReduceOp
    : TTX_Op<"reduce", [Pure, SameOperandsEncoding, SingleBlock,
                        DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Reduction using generic combination algorithm";
  let arguments = (ins Variadic<TTX_Type>:$srcs, I32Attr:$axis,
      I32Attr:$loopNum, TT_Int:$loopIndex);
  let results = (outs Variadic<TT_Type>:$result);
  let regions = (region SizedRegion<1>:$combineOp);
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let extraClassDeclaration = [{
      llvm::SmallVector<RankedTensorType> getInputTypes();
      llvm::SmallVector<Type> getElementTypes();
      unsigned getNumOperands();
    }];
}

def TTX_ReduceReturnOp : TTX_Op<"reduce.return", [HasParent<"ReduceOp">, Pure,
                                                  Terminator, ReturnLike]> {
  let summary = "terminator for reduce operator";
  let arguments = (ins Variadic<AnyType>:$result);
  let assemblyFormat = "$result attr-dict `:` type($result)";
}

// borrowed from triton/include/triton/Dialect/Triton/IR/TritonOps.td
def TTX_ScanOp
    : TTX_Op<"scan", [Pure,
                      // SameOperandsAndResultEncoding, Cause LoopIdx
                      // SameOperandsAndResultShape, Cause LoopIdx
                      SingleBlock,
                      DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Associative scan using generic combination algorithm";
  let arguments = (ins Variadic<TTX_Type>:$srcs, I32Attr:$axis,
      BoolAttr:$reverse, TT_Int:$loopIndex);
  let results = (outs Variadic<TTX_Type>:$result);
  let regions = (region SizedRegion<1>:$combineOp);
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let extraClassDeclaration = [{
      llvm::SmallVector<RankedTensorType> getInputTypes();
      llvm::SmallVector<Type> getElementTypes();
      unsigned getNumOperands();
    }];
}

def TTX_ScanReturnOp : TTX_Op<"scan.return", [HasParent<"ScanOp">, Pure,
                                              Terminator, ReturnLike]> {
  let summary = "terminator for scan operator";
  let arguments = (ins Variadic<AnyType>:$result);
  let assemblyFormat = "$result attr-dict `:` type($result)";
}

// borrowed from triton/include/triton/Dialect/Triton/IR/TritonOps.td
def TTX_BroadcastOp : TTX_Op<"broadcast", [Pure]> {
  let summary = "broadcast a tensor";

  let description = [{
      For a given tensor, broadcast changes one or more dimensions with size 1
      to a new size, e.g. tensor<1x32x1xf32> -> tensor<2x32x4xf32>.  You cannot
      change the size of a non-1 dimension.
    }];

  let arguments = (ins TTX_TensorVector:$src);

  let results = (outs TTX_TensorVector:$result);

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

  let hasCanonicalizeMethod = 1;
  let hasFolder = 1;
}

def TTX_XPUPrintOp
    : TTX_Op<"xpuprint", [MemoryEffects<[MemWrite<GlobalMemory>]>]>,
      Arguments<(ins I32:$pidx, I32:$pidy, I32:$pidz, I64:$outerIndex,
          I64:$innerIndex, I64:$ucIndex, I64:$innerBound, I64:$ucBound,
          StrAttr:$prefix, BoolAttr:$hex,
          Variadic<AnyTypeOf<[TTX_Type]>>:$args, DenseI32ArrayAttr:$isSigned)> {
  let summary = "Device-side print, as in XPU for debugging";
  let description = [{
    Compared with TT_PrintOP, the pid and index parameters are added here.
  }];
  let assemblyFormat = [{
    `(` $pidx `,` $pidy `,` $pidz `,` $outerIndex `,` $innerIndex `,` $ucIndex `,` $innerBound `,` $ucBound `)`
    `[` $prefix `,` `hex` `=` $hex `]`
    $args
    attr-dict `:`
    type($args)
  }];
}

//[FlagTree-klx-sync] ops referenced by per-op vectorization / Vectorize,
// synced from the sync source d116bdf4 (were omitted from the #992 baseline).
def TTX_VSigmoidFOp : TTX_VUnaryFOp<"vsigmoidf">;
def TTX_PackOp : TTX_Op<"pack", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "scalar tensor -> vector tensor, through an LM buffer";
    let description = [{
        Reinterprets a scalar-element tensor as a tensor of vectors of the
        same total element count. The scalar form is stored to $bufPtr and
        reloaded as whole vectors, one vector memory op per result slot.
    }];
    let arguments = (ins TT_Type:$src, Optional<TT_PtrLike>:$bufPtr);
    let results = (outs TTX_VectorLike:$result);
}

def TTX_UnpackOp : TTX_Op<"unpack", [MemoryEffects<[MemRead, MemWrite]>]> {
    let summary = "vector tensor -> scalar tensor, through an LM buffer";
    let description = [{
        Inverse of triton_xpu.pack. The vector form is stored to $bufPtr with
        one vector memory op per source slot and reloaded element-wise.
    }];
    let arguments = (ins TTX_VectorLike:$src, Optional<TT_PtrLike>:$bufPtr);
    let results = (outs TT_Type:$result);
}

//===----------------------------------------------------------------------===//
// TLE Ops — continuous DMA copy (descriptor-based)
//===----------------------------------------------------------------------===//

def TTX_TLECopyGlobalToLocalOp : TTX_Op<"tle_copy_g2l", [AttrSizedOperandSegments, MemoryEffects<[MemRead<GlobalMemory>, MemWrite]>]> {
    let summary = "TLE continuous DMA copy from global memory to local memory";
    let description = [{
      Copy a rectangular block from global memory (described by tensor_descriptor)
      to local memory buffer (memdesc). Offsets specify the tile position in GM.
      Strides are the GM tensor strides (in elements) per dimension.
      This op is the XPU equivalent of ttng.async_tma_copy_global_to_local.
    }];
    let arguments = (ins
        TT_TensorDescType:$desc,
        Variadic<I32>:$offsets,
        TTG_MemDescType:$dst_buffer,
        Variadic<I64>:$strides,
        Variadic<I32>:$shapes,
        DefaultValuedAttr<BoolAttr, "true">:$is_sync
    );
    // The per-core partition is fully driven by the tile's ClusterLayout, which
    // tritonxpu-tle-core-tiling stamps as a discardable attribute
    // "xpu.tile_layout". Absent => legacy default cluster encoding (flat
    // row-major cut, unchanged behavior).
    //
    // is_sync=false leaves the DMA in flight (the underlying gm2lm_v3 is always
    // asynchronous; is_sync only controls whether a fence follows it), so the
    // caller must tle_dma_wait before reading the destination buffer.
}

def TTX_TLELocalPtrOp : TTX_Op<"tle_local_ptr", [AttrSizedOperandSegments,
                                                 Pure]> {
    let summary = "Materialize LM pointers from buffer + multi-dim indices";
    let description = [{
      Given a local memory buffer (memdesc) and index tensors, compute a tensor
      of LM pointers: base_ptr + linearize(indices). The result is used with
      tl.load/tl.store for element-wise access to local memory.

      $loopIndex is the tile loop's induction variable, present only on a segment
      unroll control sliced: the i-th pointer of slice s is
      base_ptr + s * numElems + i. See tle_vload, which slices the same buffer
      the same way.
    }];
    let arguments = (ins
        TTG_MemDescType:$buffer,
        Variadic<TT_IntLike>:$indices,
        Optional<TT_Int>:$loopIndex
    );
    let results = (outs TT_Type:$result);
}

def TTX_TLECopyLocalToGlobalOp : TTX_Op<"tle_copy_l2g", [AttrSizedOperandSegments, MemoryEffects<[MemRead, MemWrite<GlobalMemory>]>]> {
    let summary = "TLE continuous DMA copy from local memory to global memory";
    let description = [{
      Copy a rectangular block from local memory buffer (memdesc) to global memory
      (described by tensor_descriptor). Offsets specify the tile position in GM.
      Strides are the GM tensor strides (in elements) per dimension.
      This op is the XPU equivalent of ttng.async_tma_copy_local_to_global.
    }];
    let arguments = (ins
        TT_TensorDescType:$desc,
        Variadic<I32>:$offsets,
        TTG_MemDescType:$src_buffer,
        Variadic<I64>:$strides,
        Variadic<I32>:$shapes,
        DefaultValuedAttr<BoolAttr, "true">:$is_sync
    );
    // Per-core partition driven by the tile's ClusterLayout ("xpu.tile_layout"
    // discardable attr, stamped by tritonxpu-tle-core-tiling). Absent => legacy
    // default cluster encoding (flat row-major cut).
    //
    // is_sync=false drops only the fence that WAITS for the DMA to land in GM;
    // the fence that flushes the preceding LM writes (so the DMA engine reads
    // the finished tile) is unconditional. The caller must then issue a
    // tle_dma_wait before reusing the source buffer or before the kernel ends.
}

def TTX_TLEDmaWaitOp : TTX_Op<"tle_dma_wait", []> {
    let summary = "Wait for outstanding TLE DMA transfers";
    let description = [{
      Barrier for in-flight asynchronous DMA issued by tle_copy_g2l /
      tle_copy_l2g with is_sync=false. Lowers to a single `mfence` whose mask
      selects the memory classes to drain: bit0=LM(1), bit1=SM(2), bit2=GM(4).

      Used to build software-pipelined loops: issue the copy for tile i+1 into
      the alternate buffer (is_sync=false), compute on tile i out of the current
      buffer, then tle_dma_wait before the buffers swap roles. The DMA and the
      compute overlap inside that window.
    }];
    let arguments = (ins DefaultValuedAttr<I32Attr, "5">:$mask);
    let assemblyFormat = "attr-dict";
}

def TTX_TLENormCopyGlobalToLocalOp : TTX_Op<"tle_normcopy_g2l", [MemoryEffects<[MemRead<GlobalMemory>, MemWrite]>]> {
    let summary = "TLE element-wise (permuted) gather copy from a GM pointer tensor to local memory";
    let description = [{
      Copy elements from global memory addressed by a tensor of pointers into a
      local memory buffer (memdesc). Unlike tle_copy_g2l this does NOT require an
      affine/rectangular tile: the per-element GM addresses are arbitrary (e.g. a
      permuted gather), so a TensorDescriptor cannot express it. Element i of the
      per-core lane set is copied to local slot i, matching tle_local_ptr layout.
    }];
    let arguments = (ins
        TT_Type:$src_ptrs,
        TTG_MemDescType:$dst_buffer,
        DefaultValuedAttr<BoolAttr, "true">:$is_sync
    );
}

def TTX_TLENormCopyLocalToGlobalOp : TTX_Op<"tle_normcopy_l2g", [MemoryEffects<[MemRead, MemWrite<GlobalMemory>]>]> {
    let summary = "TLE element-wise (permuted) scatter copy from local memory to a GM pointer tensor";
    let description = [{
      Copy elements from a local memory buffer (memdesc) to global memory addressed
      by a tensor of pointers. The scatter counterpart of tle_normcopy_g2l.

      `is_sync = false` drops only the TRAILING fence. The leading fence is a RAW
      hazard guard -- it makes the CPU-side LM stores visible to the DMA engine --
      and is kept unconditionally.
    }];
    let arguments = (ins
        TTG_MemDescType:$src_buffer,
        TT_Type:$dst_ptrs,
        DefaultValuedAttr<BoolAttr, "true">:$is_sync
    );
}

def TTX_TLEWaitOp : TTX_Op<"tle_wait"> {
    let summary = "Drain in-flight TLE DMA traffic in the named memory spaces.";
    let description = [{
      Bulk drain, not a counting wait: the XPU cluster ISA has no per-transfer or
      counting wait, so this drains everything outstanding in the spaces named by
      `mask` (bit0=LM, bit1=SM, bit2=GM). Pairs with an is_sync=false copy.

      The masks are NOT monotonic in their bits, so never derive one by inclusion:
      name only the LM/SM-side bit of the transfer (GM->LM and LM->GM = 1,
      GM->SM and SM->GM = 2), never the GM bit.
    }];
    let arguments = (ins DefaultValuedAttr<I32Attr, "1">:$mask);
    let assemblyFormat = "attr-dict";
}

// Vectorized local-memory load/store for the TLE compute segment. Unlike
// tt.load/tt.store, these carry NO pointee-type constraint, so their result /
// value tensor element type can be a VectorType<W x elem> produced by the
// tritonxpu-vectorize TLE branch. They read/write W contiguous elements per
// slot directly from the per-core LM buffer (no GM2LMOp / DMA involved).
//
// tle_vload declares MemRead (NOT Pure): the LM buffer it reads is refilled by
// tle_copy_g2l and overwritten by tle_vstore, so a Pure marking let CSE merge
// two reads of the same buffer across an intervening reload. That silently
// corrupted the double-buffered layernorm kernel, whose manual 2x unroll reads
// the weight buffers twice per loop body with a copy_g2l in between.
def TTX_TLEVLoadOp : TTX_Op<"tle_vload", [AttrSizedOperandSegments,
                                          MemoryEffects<[MemRead]>]> {
    let summary = "TLE vectorized load from a local-memory buffer";
    let description = [{
      Load a per-core local-memory buffer as a tensor whose element type is a
      VectorType<W x elem>. Element i maps to
      buffer[(base + i)*W : (base + i + 1)*W], where `base` comes from the
      optional `$loopIndex` (0 when absent).

      `$loopIndex` is named and used the way xpu::MakeRangeOp, xpu::LoadOp,
      xpu::ReduceOp and xpu::ScanOp carry theirs -- the tile loop's induction
      variable, put there by unroll control when it slices a segment, so
      base = loopIndex * numVecs; without it every iteration would re-read
      slice 0. Applies to both scopes.

      `$smElemOffset` is the separate, cluster-shared-buffer quantity: the
      uniform ELEMENT addend the `tle_local_ptr` index tensor carried
      (`[roff +] arange(0, n)`, admitted by `tleSmemSliceOffset`). It is its own
      operand rather than sharing `$loopIndex` because a scope=smem read can be
      BOTH sliced by unroll control AND offset by its index: one operand for two
      quantities got them added as if they were one number and read another
      core's columns (silent max_diff 8.69 on layernorm 1024x4096 XB16 RI4).
      Absent for an LM read, whose index tensor the lowering ignores.
    }];
    let arguments = (ins TTG_MemDescType:$buffer,
                         Optional<TT_Int>:$loopIndex,
                         Optional<TT_Int>:$smElemOffset);
    let results = (outs TT_Type:$result);
}

def TTX_TLEVStoreOp : TTX_Op<"tle_vstore"> {
    let summary = "TLE vectorized store to a local-memory buffer";
    let description = [{
      Store a tensor whose element type is a VectorType<W x elem> back into a
      per-core local-memory buffer. Element i maps to
      LM[(loopIndex * numVecs + i)*W : (loopIndex * numVecs + i + 1)*W]; see
      tle_vload for $loopIndex.
    }];
    let arguments = (ins TTG_MemDescType:$buffer, TT_Type:$value,
                         Optional<TT_Int>:$loopIndex);
}

//===----------------------------------------------------------------------===//
// tle.raw payload injection on the cluster path.
//===----------------------------------------------------------------------===//

def TTX_RawOp : TTX_Op<"raw", []> {
    let summary = "call a user-provided LLVM IR payload from a cluster kernel";

    let description = [{
      `triton_xpu.raw` calls the function `$callee`, which `tle.raw.call` names by
      its payload's `$source_id`: the payload is LLVM 19 IR text that the backend
      compiles for the arch it is building for and merges into the kernel module
      at the LLVM 19 stage (see `tle/raw/merge.py`), after TritonXPU->LLVM
      conversion has turned this op into an `always_inline` call.

      This is the xpu3 cluster counterpart of `sdnn.raw`, and mirrors the
      alternate cluster path's raw op. The cluster path never sees
      memrefs, so the calling convention is much simpler: every operand is
      passed through as-is after type conversion -- `tt.ptr<T>` becomes an
      `!llvm.ptr` in the global address space and scalars stay scalars. Tensor
      operands are not supported: a tensor lives in per-core registers/LM here,
      so there is no address to hand over -- stage the data through pointers.

      Example:
      ```mlir
      triton_xpu.raw "my_add"(%out, %in, %n)
        {source_id = "8f14e45fceea167a5a36dedd4bea2543"}
        : (!tt.ptr<f32>, !tt.ptr<f32>, i32) -> ()
      ```

      The op has no results and no declared memory effects, so it is never
      speculated or dead-code eliminated: the payload is assumed to write
      through the pointers it is handed.
    }];

    let arguments = (ins
        StrAttr:$callee,
        StrAttr:$source_id,
        Variadic<AnyType>:$args
    );

    let assemblyFormat = [{
      $callee `(` $args `)` attr-dict `:` functional-type($args, results)
    }];
}

#endif // TRITONXPU_OPS
