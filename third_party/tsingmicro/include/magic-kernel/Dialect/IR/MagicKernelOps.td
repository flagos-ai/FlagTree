//===------------------- MagicKernelOps.td --------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// The abstract layer between MLIR core dialects and the lower target specific
// dialects of MagicKernelFunc and MagicKernelInstr.
//
// Compare to higher level MLIR dialects such as memref, arith, affine etc, the
// granularity of MK dialect is more suitable to map into ML accelerators. For
// example, tt.load is lowered to arith + memref.reinterpret_cast + memref.alloc
// + memref.copy + bufferization.to_tensor by decoding hidden high level info
// into detailed info carried in those core MLIR dialects.
// If we convert tt.load to mk.alloc + mk.load, we have to redo all the analysis
// and info constructions which triton-shared already does, so that we should
// generate mk.alloc + mk.load from the core dialects to avoid reconstructing
// the information.
// By doing so, we can lower arith + memref.reinterpret_cast + memref.copy +
// buf.to_tensor into mk.load, and lower arith + memref.alloc into mk.alloc.
//
//===----------------------------------------------------------------------===//

#ifndef MAGIC_KERNEL_OPS
#define MAGIC_KERNEL_OPS

include "magic-kernel/Dialect/IR/MagicKernelTypes.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultType
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// Bufferable type.
//===----------------------------------------------------------------------===//

def TensorOrMemref :
  AnyTypeOf<[AnyMemRef, AnyRankedTensor], "", "::mlir::ShapedType">;

def I1TensorOrMemref :
  AnyTypeOf<[MemRefOf<[I1]>, RankedTensorOf<[I1]>], "", "::mlir::ShapedType">;

def IntTensorOrMemref :
  AnyTypeOf<[MemRefOf<[AnyInteger]>, RankedTensorOf<[AnyInteger]>], "", "::mlir::ShapedType">;

def FPTensorOrMemref :
  AnyTypeOf<[MemRefOf<[AnyFloat]>, RankedTensorOf<[AnyFloat]>], "", "::mlir::ShapedType">;

//
// Interfaces
//
def GlobalMemory : Resource<"::mlir::triton::GlobalMemory">;


class MKOp<string mnemonic, list<Trait> traits = []> :
  Op<MagicKernelDialect, mnemonic,
     !listconcat(traits, [/*TensorSizeTrait, VerifyTensorLayoutsTrait*/])> {
}

class MKUnElemWiseOp<string mnemonic> : MKOp<mnemonic, [Elementwise,
                                             DestinationStyleOpInterface]> {
  let summary = "Element wise unary operation: $mnemonic";

  let arguments = (
    ins
    TensorOrMemref:$src,
    // buffer for store result
    Arg<TensorOrMemref, "the target memref", [MemWrite]>:$zeroes,
    BoolAttr:$is_atomic
  );

  let results = (outs Variadic<TensorOrMemref>:$dst);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getZeroesMutable();
    }
  }];

}

class MKBinElemWiseOp<string mnemonic> : MKOp<mnemonic, [Pure, Elementwise]> {
  let summary = "Element wise binary operation: $mnemonic";

  let arguments = (
    ins
    AnyTensor:$src0,
    AnyTensor:$src1,
    BoolAttr:$is_atomic
  );

  let results = (outs AnyTensor:$dst);
}

class MKTerElemWiseOp<string mnemonic> : MKOp<mnemonic, [Pure, Elementwise]> {
  let summary = "Element wise binary operation: $mnemonic";

  let arguments = (
    ins
    AnyTensor:$src0,
    AnyTensor:$src1,
    AnyTensor:$src2,
    BoolAttr:$is_atomic
  );

  let results = (outs AnyTensor:$dst);
}

// arith.bitcast doesn't support pointers
def BitcastOp : MKOp<"bitcast", [Pure]> {
    let summary = "Cast between types of the same bitwidth";

    let arguments = (ins AnyType:$src);

    let results = (outs AnyType:$result);

    let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";

    let hasFolder = 1;

    // TODO: Add verifier
}

// =============================================================================
// Memory allocation ops
// =============================================================================

def AllocOp : MKOp<"alloc", []> {
  let summary = "Allocate a consecutive memory from given addressing space";

  let description = [{
    It may or may not generate target intrinsic call or instruction, the
    lowering from this operator to lower level operator is target specific.
  }];

  let arguments = (
    ins
    I32Attr:$addr_space,  // The addressing space
    I64ArrayAttr:$dims    // The size of memory to be allocated
  );

  // Return the pointer of the allocated memory
  let results = (outs AnyRankedOrUnrankedMemRef:$ptr);
}

// =============================================================================
// Load/Store Ops
// =============================================================================

// Unit and strided memory load
def LoadOp : MKOp<"load", []> {
  let summary = "Load from a memory with optional strides";

  let description = [{ See RISC-V RVV unit/strided memory load for detail }];

  let arguments = (
    ins
    AnyRankedOrUnrankedMemRef:$ptr, // The base ptr
    I32Attr:$addr_space,            // The address space
    I64ArrayAttr:$dims,             // The shape to be loaded, can be dynamic
    I64ArrayAttr:$strides,          // The strides in each rank, can be dynamic
    BoolAttr:$mask        // element is not loaded if mask[i] == 0
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $ptr `,` attr-dict `:` type($ptr) `->` type($result)
  }];
}

// Index memory load
def IndexLoadOp : MKOp<"iload", [
]> {
  let summary = "Load from a memory with indexed offset";

  let description = [{ See RISC-V RVV index memory load for detail }];

  let arguments = (
    ins
    AnyRankedOrUnrankedMemRef:$ptr, // The base ptr
    I32Attr:$addr_space,            // The address space
    I64ArrayAttr:$dims,             // The shape to be loaded
    AnyTensor:$index,    // The tensor contains memory offset for each element
    BoolAttr:$mask      // element is not loaded if mask[i] == 0
  );

  let results = (outs MKType:$result);
}

// Unit and strided memory store
def StoreOp : MKOp<"store", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Store to a memory with optional strides";

  let description = [{ See RISC-V RVV unit/strided memory store for detail }];

  let arguments = (
    ins
    AnyRankedOrUnrankedMemRef:$ptr, // The base ptr
    I32Attr:$addr_space,            // The address space
    I64ArrayAttr:$dims,             // The shape to be stored
    I64ArrayAttr:$strides,          // The strides in each rank
    BoolAttr:$mask // element is not write to dest if mask[i] == 0
  );

  let assemblyFormat = [{
    $ptr `,` attr-dict `:` type($ptr)
  }];
}

// Index memory store
def IndexStoreOp : MKOp<"istore", [MemoryEffects<[MemWrite<GlobalMemory>]>]> {
  let summary = "Store to a memory with indexed offset";

  let description = [{ See RISC-V RVV index memory store for detail }];

  let arguments = (
    ins
    AnyRankedOrUnrankedMemRef:$ptr, // The base ptr
    I32Attr:$addr_space,            // The address space
    I64ArrayAttr:$dims,             // The shape to be stored
    AnyTensor:$index,     // The tensor contains memory offset for each element
    BoolAttr:$mask        // element is not write to dest if mask[i] == 0
  );
}

// =============================================================================
// DataMove
// =============================================================================

def MaskMoveOp : MKOp<"mask_move", [DestinationStyleOpInterface]> {
  let summary = "Mask data move API";

  let description = [{ When mask is 1, extract the data from src and write it to dst.
When mask=0, the corresponding elements of dst remain unchanged.
  }];

  let arguments = (
    ins
    TensorOrMemref:$source,            // The source address in SPM
    TensorOrMemref:$mask,
    Arg<TensorOrMemref, "the init memref", [MemWrite]>:$init     // The init buffer
  );

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getInitMutable();
    }
  }];

  // The dst address is not used, use init instead.
  let results = (outs Variadic<TensorOrMemref>:$dst);
}

def GatherScatter : MKOp<"gatherscatter", []> {
  let summary = "Transfer data in strides and iterations";

  let arguments = (ins
    TensorOrMemref:$source,                 // The source
    TensorOrMemref:$target,                 // The target
    I32Attr:$bytes,                  // Inner loop data size in bytes
    I32Attr:$src_strideN,
    I32Attr:$src_strideH,
    I32Attr:$src_strideW,
    I32Attr:$src_iterN,
    I32Attr:$src_iterH,
    I32Attr:$src_iterW,
    I32Attr:$dst_strideN,
    I32Attr:$dst_strideH,
    I32Attr:$dst_strideW,
    I32Attr:$dst_iterN,
    I32Attr:$dst_iterH,
    I32Attr:$dst_iterW
  );
  let results = (outs Variadic<TensorOrMemref>:$dst);
}

// =============================================================================
// Dot op
// =============================================================================

def DotOp : MKOp<"dot", [DestinationStyleOpInterface]> {
  let summary = "Inner production of 2 vectors";

  let description = [{
    TODO: It is currently one to one mapping from upper dialect tt.dot.
  }];

  let arguments = (
    ins
    TensorOrMemref:$a,           // Matrix A
    TensorOrMemref:$b,           // Matrix B
    Arg<TensorOrMemref, "the target memref", [MemWrite]>:$inits,
    BoolAttr:$en_psum            // Enable psum. Used as accumulate buffer
  );

  let results = (outs Variadic<TensorOrMemref>:$d);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getInitsMutable();
    }
  }];

  // let hasVerifier = 1;
}


// =============================================================================
// Dot Scaled op
// =============================================================================

// Type for ScaleDotElemType kind of floats.
def MK_ScaleDotElemTypeAttr : I32EnumAttr<
    "ScaleDotElemType", "",
    [
      I32EnumAttrCase<"E4M3", 0, "e4m3">,
      I32EnumAttrCase<"E5M2", 1, "e5m2">,
      I32EnumAttrCase<"E2M3", 2, "e2m3">,
      I32EnumAttrCase<"E3M2", 3, "e3m2">,
      I32EnumAttrCase<"E2M1", 4, "e2m1">,
      I32EnumAttrCase<"BF16", 5, "bf16">,
      I32EnumAttrCase<"FP16", 6, "fp16">
    ]>{
  let cppNamespace = "::mlir::triton";
}

def DotScaledOp : MKOp<"dot_scaled", [DestinationStyleOpInterface, AttrSizedOperandSegments]> {
    let summary = "dot_scaled";

    let description = [{
        $dst = matrix_multiply(scale($a, $a_scale), scale($b, $b_scale)).
        Where scale(x, s) is a function that applies the scale per block following microscaling spec.
    }];

    let arguments = (
      ins
      // inputs are floats if we have a type for them, otherwise (fp4),
      // they are packed in pairs in an I8Tensor
      TensorOrMemref:$a,
      Optional<TensorOrMemref>:$a_scale,
      TensorOrMemref:$b,
      Optional<TensorOrMemref>:$b_scale,
      Arg<TensorOrMemref, "the target memref", [MemWrite]>:$dst,
      MK_ScaleDotElemTypeAttr:$a_elem_type,
      MK_ScaleDotElemTypeAttr:$b_elem_type,
      BoolAttr:$fastMath
    );

    let results = (outs Variadic<TensorOrMemref>:$res);

      let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}

// =============================================================================
// Reduction ops
// =============================================================================
class ReduceOp<string mnemonic> : MKOp<mnemonic, [DestinationStyleOpInterface]> {
  let summary = "Compute extremal values and their indices from source tensor.";

  let arguments = (
    ins
    TensorOrMemref:$src,
    Arg<TensorOrMemref, "the target memref", [MemWrite]>:$init,
    I64ArrayAttr: $nhwcShape,
    I32Attr:$axis
  );

  let results = (outs Variadic<TensorOrMemref>:$result);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getInitMutable();
    }
  }];
}

// TODO: Better way to define reduce op
// A complementary operation for linalg.reduce since tsing-micro need channel-norm
def ReduceMaxOp : ReduceOp<"reduce_max"> {}
def ReduceMinOp : ReduceOp<"reduce_min"> {}
def ReduceSumOp : ReduceOp<"reduce_sum"> {}
def XorSumOp : MKOp<"xor_sum"> {}

class ArgReduceOp<string mnemonic> : MKOp<mnemonic, [DestinationStyleOpInterface]> {
  let summary = "Compute extremal values and their indices from source tensor.";

  let arguments = (
    ins
    TensorOrMemref:$src,
    Arg<TensorOrMemref, "the target memref", [MemWrite]>:$value,
    Arg<TensorOrMemref, "the target memref", [MemWrite]>:$index,
    I32Attr:$axis
  );

  let results = (outs Variadic<TensorOrMemref>:$result);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return MutableOperandRange(getOperation(), /*start=*/1, /*count=*/2);
    }
  }];
}

def ArgMaxOp : ArgReduceOp<"argmax"> {}
def ArgMinOp : ArgReduceOp<"argmin"> {}


// =============================================================================
// Scan/Sort Ops
// =============================================================================

def SortOp : MKOp<"sort", [Pure]> {}
def GatherOp : MKOp<"gather", [DestinationStyleOpInterface]> {
  let summary = "Gather from a tensor along a given dimension.";

  let description = [{
    TODO: It is currently one to one mapping from upper dialect tt.gather.
  }];

  let arguments = (
    ins
    TensorOrMemref:$src,               // input
    TensorOrMemref:$indices,           // indices
    Arg<TensorOrMemref, "the target memref", [MemWrite]>:$dst, // output
    I32Attr:$axis
  );

  let results = (outs Variadic<TensorOrMemref>:$result);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}


// =============================================================================
// Unary/Binary/Ternary Element-wise Math Ops
// =============================================================================

def AbsOp : MKUnElemWiseOp<"abs">;
def AddOp : MKBinElemWiseOp<"add">;
def AndOp : MKBinElemWiseOp<"and">;
def CDivOp : MKBinElemWiseOp<"cdiv">;
def CeilOp : MKUnElemWiseOp<"ceil">;
def ClampOp : MKUnElemWiseOp<"clamp">;
def CosOp : MKUnElemWiseOp<"cos">;
def DivOp : MKBinElemWiseOp<"div">;
def ErfOp : MKUnElemWiseOp<"erf">;
def ExpOp : MKUnElemWiseOp<"exp">;
def Exp2Op : MKUnElemWiseOp<"exp2">;
def FdivOp : MKBinElemWiseOp<"fdiv">;
def FloorOp : MKUnElemWiseOp<"floor">;
def FmaOp : MKTerElemWiseOp<"fma">;
def LogOp : MKUnElemWiseOp<"log">;
def Log2Op : MKUnElemWiseOp<"log2">;
def MaxOp : MKUnElemWiseOp<"max">;
def MinOp : MKUnElemWiseOp<"min">;
def OrOp : MKBinElemWiseOp<"or">;
def RsqrtOp : MKUnElemWiseOp<"rsqrt">;
def SigmoidOp : MKUnElemWiseOp<"sigmoid">;

def GeluOp : MKOp<"gelu", [DestinationStyleOpInterface]> {
  let summary = "Fusion gelu op";

  let arguments = (
    ins
    TensorOrMemref:$src,
    Optional<TensorOrMemref>:$imm, // WORKAROUND for tx backend
    // buffer for store result
    Arg<TensorOrMemref, "the target memref", [MemWrite]>:$zeroes,
    BoolAttr:$is_atomic,
    I16Attr:$gelu_mode
  );

  let results = (outs Variadic<TensorOrMemref>:$dst);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getZeroesMutable();
    }
  }];

}
def SinOp : MKUnElemWiseOp<"sin">;
def SqrtOp : MKUnElemWiseOp<"sqrt">;
def SqrtRnOp : MKUnElemWiseOp<"sqrt_rn">;
def XorOp : MKBinElemWiseOp<"xor">;
// def UmulhiOp : MKOp<"umulhi", [Pure]> {}

def BarrierOp : MKOp<"barrier"> {
  let summary = "Synchronizes all work items";
  let description = [{
    The "barrier" op synchronizes all work items.
  }];
  let assemblyFormat = "attr-dict";
}

def PrintOp : MKOp<"print", [MemoryEffects<[MemWrite]>, DestinationStyleOpInterface]> {
  let summary = "Print at most a single scalar or 1D TensorOrMemref on each line";

  let description = [{
    It only takes a single scalar or 1D TensorOrMemref element.
  }];

  let arguments = (ins
    StrAttr:$prefix,
    BoolAttr:$hex,
    Variadic<AnyTypeOf<[MKFloat, MKInt, MKPtr, TensorOrMemref]>>:$val,
    DenseI32ArrayAttr:$isSigned
  );

  let results = (outs Variadic<TensorOrMemref>:$dst);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getValMutable();
    }
  }];

  let hasVerifier = 1;
}

def AssertOp : MKOp<"assert"> {
  let summary = "Assert the condition at runtime from the device";

  let arguments = (ins
    StrAttr:$message
  );
}


// =============================================================================
// Binary(scalr and tensor) Element-wise Math Ops
// =============================================================================

class ArithVSOp<string mnemonic, list<Trait> traits = [DestinationStyleOpInterface]> :
    MKOp<mnemonic, traits # [Elementwise]> {

  let summary = "Vector-Scalar arith op which return output with same input vector type";

  let arguments = (ins
    FPTensorOrMemref:$input,              // First input vector address
    AnyFloat:$value,                       // Const value
    Arg<TensorOrMemref, "the init memref", [MemWrite]>:$init  // Out vector address
  );
  let results = (outs Variadic<TensorOrMemref>:$dst);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getInitMutable();
    }
  }];
}

def AddVS : ArithVSOp<"addvs">;
def SubVS : ArithVSOp<"subvs">;
def MulVS : ArithVSOp<"mulvs">;

// =============================================================================
// Relation Ops
// =============================================================================
class RelationVVOp<string mnemonic, list<Trait> traits = [DestinationStyleOpInterface]> :
    MKOp<mnemonic, traits # [Elementwise]> {
  let summary = "Vector-Vector relation op which return output with same input type";

  let arguments = (ins
    FPTensorOrMemref:$input0,              // First input vector address
    FPTensorOrMemref:$input1,              // Second vector address
    Arg<FPTensorOrMemref, "the init memref", [MemWrite]>:$init  // init vector address
  );
  let results = (outs Variadic<FPTensorOrMemref>:$dst);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getInitMutable();
    }
  }];
}

def EqualVV : RelationVVOp<"equalvv"> {
  let summary = "compare two input value, if equal, return 1.0";
}

def UnEqualVV : RelationVVOp<"unequalvv"> {
  let summary = "compare two input value, if unequal, return 1.0";
}

def GreaterEqualVV : RelationVVOp<"greatrequalvv"> {
  let summary = "compare two input value, if src0 >= src1, return 1.0";
}

def GreaterVV : RelationVVOp<"greatervv"> {
  let summary = "compare two input value, if src0 > src1, return 1.0";
}

def LessEqualVV : RelationVVOp<"lessequalvv"> {
  let summary = "compare two input value, if src0 <= src1, return 1.0";
}

def LessThenVV : RelationVVOp<"lessthenvv"> {
  let summary = "compare two input value, if src0 < src1, return 1.0";
}

class RelationVSOp<string mnemonic, list<Trait> traits = [DestinationStyleOpInterface]> :
    MKOp<mnemonic, traits # [Elementwise]> {

  let summary = "Vector-Scalar relation op which return output with same input vector type";

  let arguments = (ins
    FPTensorOrMemref:$input,              // First input vector address
    AnyFloat:$value,                       // Const value
    Arg<TensorOrMemref, "the init memref", [MemWrite]>:$init  // Out vector address
  );
  let results = (outs Variadic<TensorOrMemref>:$dst);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getInitMutable();
    }
  }];
}

def BoolEqualVS : RelationVSOp<"boolequalvs"> {
  let summary = "compare input value with ConstantOp, if equal, return true";
}

def BoolUnEqualVS : RelationVSOp<"boolunequalvs"> {
  let summary = "compare input value with ConstantOp, if unequal, return true";
}

def BoolGreaterEqualVS : RelationVSOp<"boolgreatrequalvs"> {
  let summary = "compare input value with ConstantOp, if src0 >= src1, return true";
}

def BoolGreaterVS : RelationVSOp<"boolgreatervs"> {
  let summary = "compare input value with ConstantOp, if src0 > src1, return true";
}

def BoolLessEqualVS : RelationVSOp<"boollessequalvs"> {
  let summary = "compare input value with ConstantOp, if src0 <= src1, return true";
}

def BoolLessThenVS : RelationVSOp<"boollessthenvs"> {
  let summary = "compare input value with ConstantOp, if src0 < src1, return true";
}

def EqualVS : RelationVSOp<"equalvs"> {
  let summary = "compare input value with ConstantOp, if equal, return 1.0";
}

def UnEqualVS : RelationVSOp<"unequalvs"> {
  let summary = "compare input value with ConstantOp, if unequal, return 1.0";
}

def GreaterEqualVS : RelationVSOp<"greatrequalvs"> {
  let summary = "compare input value with ConstantOp, if src0 >= src1, return 1.0";
}

def GreaterVS : RelationVSOp<"greatervs"> {
  let summary = "compare input value with ConstantOp, if src0 > src1, return 1.0";
}

def LessEqualVS : RelationVSOp<"lessequalvs"> {
  let summary = "compare input value with ConstantOp, if src0 <= src1, return 1.0";
}

def LessThenVS : RelationVSOp<"lessthenvs"> {
  let summary = "compare input value with ConstantOp, if src0 < src1, return 1.0";
}

// =============================================================================
// Atomic Ops
// =============================================================================

def AtomicRMWOp : MKOp<"atomic_rmw", [
  DestinationStyleOpInterface
]> {
  let summary = "perform atomic read-modify-write operation on a pointer";

  let description = [{
      load data at $ptr, do $rmw_op with $val, and store result to $ptr.

      return old value at $ptr
  }];

  let arguments = (ins  Arg<AnyMemRef, "the source memref",[MemRead, MemWrite]>:$ptr,
                        AnyType:$val,
                        Arg<TensorOrMemref, "the target memref", [MemWrite]>:$dst,
                        TT_AtomicRMWAttr:$atomic_rmw_op,
                        TT_MemSemanticAttr:$sem,
                        TT_MemSyncScopeAttr:$scope);

  let results = (outs Variadic<AnyType>:$result);

  // Explicitly list $atomic_rmw_op, $sem, and $scope rather than relying on
  // attr-dict so they're printed as strings rather than opaque integers.
  let assemblyFormat = [{
    $atomic_rmw_op `,` $sem `,` $scope `,` $ptr `,` $val `,` $dst attr-dict `:`
    functional-type(operands, $result)
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}

def AtomicCASOp : MKOp<"atomic_cas", [
  DestinationStyleOpInterface
]> {
  let summary = "perform atomic compare-and-swap operation on a pointer";

  let description = [{
    compare $cmp with data $old at location $ptr,

    if $old == $cmp, store $val to $ptr,

    else store $old to $ptr,

    return $old
  }];


  let arguments = (ins  Arg<AnyMemRef, "the source memref",[MemRead, MemWrite]>:$ptr,
                        AnyType:$cmp, AnyType:$val,
                        Arg<TensorOrMemref, "the target memref", [MemWrite]>:$dst,
                        TT_MemSemanticAttr:$sem,
                        TT_MemSyncScopeAttr:$scope);

  let results = (outs Variadic<AnyType>:$result);

  // Explicitly list $sem and $scope rather than relying on attr-dict so
  // they're printed as strings rather than opaque integers.
  let assemblyFormat = [{
    $sem `,` $scope `,` $ptr `,` $cmp `,` $val `,` $dst attr-dict `:`
    functional-type(operands, $result)
  }];

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getDstMutable();
    }
  }];
}

// =============================================================================
// Sync ops
// =============================================================================
def AtomicBarrierInOp : MKOp<"atomic_barrier_in", [MemoryEffects<[MemRead, MemAlloc]>]> {
  let summary = "Synchronizes all work items before input";
  let description = [{
    The "barrier in" op synchronizes all work items.
  }];
  let assemblyFormat = "attr-dict";
}

def AtomicBarrierOutOp : MKOp<"atomic_barrier_out", [MemoryEffects<[MemAlloc, MemWrite]>]> {
  let summary = "Synchronizes all work items after output";
  let description = [{
    The "barrier out" op synchronizes all work items.
  }];
  let assemblyFormat = "attr-dict";
}

// =============================================================================
// Peripheral instructions
// =============================================================================
def Bit2FpOp : MKOp<"bit2fp", [DestinationStyleOpInterface]> {
  let summary = "Convert a vector of the bitwise into the fp vector";

  let arguments = (ins
    I1TensorOrMemref:$src,             // Input tensor
    Arg<FPTensorOrMemref, "the init memref", [MemWrite]>:$init
  );
  let results = (outs Variadic<FPTensorOrMemref>:$result);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getInitMutable();
    }
  }];
}

def CastOp : MKOp<"cast", [DestinationStyleOpInterface]> {
  let summary = "Cast a tensor or memref to a different type";

  let arguments = (ins
    TensorOrMemref:$src,             // Input tensor
    Arg<TensorOrMemref, "the init memref", [MemWrite]>:$init
  );
  let results = (outs Variadic<TensorOrMemref>:$result);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getInitMutable();
    }
  }];
}

// TODO: May be replaced by quant dialect ops in the future
def DequantOp : MKOp<"dequant", [DestinationStyleOpInterface]> {
  let summary = "output = (input - zero_point) * scale";

  let arguments = (ins
    TensorOrMemref:$src,             // Input tensor
    TensorOrMemref:$scale,              // scale: float but stored as uint8
    Arg<FPTensorOrMemref, "the init memref", [MemWrite]>:$init
  );
  let results = (outs Variadic<FPTensorOrMemref>:$result);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getInitMutable();
    }
  }];
}

#endif // MAGIC_KERNEL_OPS
