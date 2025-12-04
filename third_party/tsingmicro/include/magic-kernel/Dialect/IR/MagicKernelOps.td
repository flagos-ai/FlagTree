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
def BitcastOp : MKOp<"bitcast", [Elementwise, Pure]> {
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
    Optional<TensorOrMemref>:$c, // Optional accumulation matrix C
    // Zeroes buffer which can be used to fill $d
    // FIXME: Whether need add side effect to source operands?
    Arg<TensorOrMemref, "the target memref", [MemWrite]>:$zeroes
    //DefaultValuedAttr<TT_InputPrecisionAttr,
    //                  "::mlir::triton::InputPrecision::IEEE">:$inputPrecision,
    // DefaultValuedAttr<I32Attr, "0">:$maxNumImpreciseAcc
  );

  let results = (outs Variadic<TensorOrMemref>:$d);

  let extraClassDeclaration = [{
    MutableOperandRange getDpsInitsMutable() {
      return getZeroesMutable();
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

def ArgMaxOp : MKOp<"argmax", [Pure]> {}
def ArgMinOp : MKOp<"argmin", [Pure]> {}
def ReduceMaxOp : MKOp<"reduce_max", [Pure]> {}
def ReduceMinOp : MKOp<"reduce_min", [Pure]> {}
def ReduceOp : MKOp<"reduce", [Pure]> {}
def SumOp : MKOp<"sum", [Pure]> {}
def XorSumOp : MKOp<"xor_sum", [Pure]> {}


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

def PrintOp : MKOp<"print", [DestinationStyleOpInterface]> {
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

#endif // MAGIC_KERNEL_OPS
