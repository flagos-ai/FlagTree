// Copyright 2026- Xcoresigma Technology Co., Ltd

#ifndef TRITON_TLE_DSA_OPS
#define TRITON_TLE_DSA_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "triton/Dialect/Triton/IR/TritonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "tle/dsa/dialect/include/IR/TleDSADialect.td"
include "tle/dsa/dialect/include/IR/TleDSAAttrDefs.td"

//
// Op Base
//
class TLE_Op<string mnemonic, list<Trait> traits = []>:
    Op<Tle_Dialect, mnemonic,
        !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

//
// Copy OP
//
def TLE_DSACopyOp : TLE_Op<"dsa_copy", [Pure, MemoryEffects<[MemWrite]>]> {
  let summary = "self-defined copy operation";
  let description = [{
    'tle.dsa_copy' triton copy op is designed to copy data between memory regions.
    Example:
    ```mlir
    tle.dsa_copy %src, %dst, %shape : tensor<128xf32>
    ```
  }];
  let arguments = (ins
    AnyType:$src,
    AnyType:$dst,
    Variadic<I32>:$shape
  );

  //assemble
  let assemblyFormat = "$src `,`  $dst `,` $shape attr-dict `:` type($src) `,` type($dst) `,` `[`type($shape)`]`";
}

//
// Add Op
//
def TLE_DSAAddOp : TLE_Op<"dsa_add", [Pure, MemoryEffects<[MemWrite]>,
                                      SameOperandsShape, SameOperandsElementType]> {
  let summary = "self-defined add operation";
  let description = [{
    `tle.dsa_add` triton dsa_add op is designed to performs element-wise addition.
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$res);

  let assemblyFormat = "$lhs `,` $rhs `,` $res attr-dict `:` type($lhs) `,` type($rhs) `,` type($res)";
}

//
// Sub op
//
def TLE_DSASubOp : TLE_Op<"dsa_sub", [Pure, MemoryEffects<[MemWrite]>,
                                      SameOperandsShape, SameOperandsElementType]> {
  let summary = "self-defined sub operation";
  let description = [{
    `tle.dsa_sub` triton dsa_sub op is designed to performs element-wise addition.
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$res);

  let assemblyFormat = "$lhs `,` $rhs `,` $res attr-dict `:` type($lhs) `,` type($rhs) `,` type($res)";
}

//
// Mul op
//
def TLE_DSAMulOp : TLE_Op<"dsa_mul", [Pure, MemoryEffects<[MemWrite]>,
                                      SameOperandsShape, SameOperandsElementType]> {
  let summary = "self-defined mul operation";
  let description = [{
    `tle.dsa_mul` triton dsa_mul op is designed to performs element-wise addition.
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$res);

  let assemblyFormat = "$lhs `,` $rhs `,` $res attr-dict `:` type($lhs) `,` type($rhs) `,` type($res)";
}

//
// Div op
//
def TLE_DSADivOp : TLE_Op<"dsa_div", [Pure, MemoryEffects<[MemWrite]>,
                                      SameOperandsShape, SameOperandsElementType]> {
  let summary = "self-defined div operation";
  let description = [{
    `tle.dsa_div` triton dsa_div op is designed to performs element-wise addition.
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$res);

  let assemblyFormat = "$lhs `,` $rhs `,` $res attr-dict `:` type($lhs) `,` type($rhs) `,` type($res)";
}

//
// Max op
//
def TLE_DSAMaxOp : TLE_Op<"dsa_max", [Pure, MemoryEffects<[MemWrite]>,
                                      SameOperandsShape, SameOperandsElementType]> {
  let summary = "self-defined Max operation";
  let description = [{
    `tle.dsa_max` triton dsa_max op is designed to performs element-wise addition.
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$res);

  let assemblyFormat = "$lhs `,` $rhs `,` $res attr-dict `:` type($lhs) `,` type($rhs) `,` type($res)";
}

//
// Min op
//
def TLE_DSAMinOp : TLE_Op<"dsa_min", [Pure, MemoryEffects<[MemWrite]>,
                                      SameOperandsShape, SameOperandsElementType]> {
  let summary = "self-defined Min operation";
  let description = [{
    `tle.dsa_min` triton dsa_min op is designed to performs element-wise addition.
  }];
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$res);

  let assemblyFormat = "$lhs `,` $rhs `,` $res attr-dict `:` type($lhs) `,` type($rhs) `,` type($res)";
}

/// //
/// // Dot op
/// //
/// def TLE_DSADotOp : TLE_Op<"dsa_dot", [Pure,
///                                     MemoryEffects<[MemWrite]>,
///                                     SameOperandsElementType,
///                                     DotLike]> {
///   let summary = "self-defined Dot operation";
///   let description = [{
///       $d = matrix_multiply($a, $b) + $c.
///   }];
///
///   let arguments = (
///     ins
///     AnyMemRef:$inA,
///     AnyMemRef:$inB,
///     AnyMemRef:$res,
///     I64ArrayAttr:$size,
///     DefaultValuedAttr<BoolAttr, "false">:$initC,
///     DefaultValuedAttr<BoolAttr, "false">:$traA,
///     DefaultValuedAttr<BoolAttr, "false">:$traB,
///     DefaultValuedAttr<BoolAttr, "true">:$enableHf32
///   );
///
///   let assemblyFormat = "$inA `,` $inB `,` $res attr-dict `:` type($inA) `,` type($inB) `,` type($res)";
/// }
def TLE_GetRankOp : TLE_Op<"get_rank", [Pure]> {
  let summary = "Rank of the calling PE";

  let arguments = (ins I32:$axis);

  let results = (outs I32:$result);

  let assemblyFormat = "$axis attr-dict `:` type($result)";
}

def TLE_GetNumRanksOp : TLE_Op<"get_num_ranks", [Pure]> {
  let summary = "Get the number of PEs runing in the program";

  let arguments = (ins I32:$axis);

  let results = (outs I32:$result);

  let assemblyFormat = "$axis attr-dict `:` type($result)";
}

def TLE_SymmAtOp : TLE_Op<"symm_at", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TypesMatchWith<"remoteAddr matches symmAddr type", "symmAddr", "remoteAddr", "$_self">,
]> {
  let summary = "obtain the symmetric address on the specified Rank.";

  let arguments = (ins
    TT_Ptr:$symmAddr,
    I32:$rank
  );

  let results = (outs TT_Ptr:$remoteAddr);

  let assemblyFormat = "$symmAddr `,` $rank `,` attr-dict `:` qualified(type($symmAddr))";
}


def TLE_ExternCallOp : TLE_Op<"extern_call", [
   DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {

    let description = [{
        call an external function $symbol implemented in $libpath/$libname with $args
        return $libpath/$libname:$symbol($args...)
    }];

    let arguments = (ins Variadic<TT_Type>:$srcs, StrAttr:$libname, StrAttr:$libpath, StrAttr:$symbol, BoolAttr:$pure);

    let results = (outs Variadic<TT_Type>:$result);

    let assemblyFormat = "operands attr-dict `:` functional-type(operands, $result)";

    let extraClassDeclaration = [{
      // Interface method for ConditionallySpeculatable.
      Speculation::Speculatability getSpeculatability();
    }];
}

def TLE_WaitOp : TLE_Op<"wait", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  TypesMatchWith<"result matches ptr type", "barrierPtr", "waitValue", "getPointeeType($_self)">,
]> {
  let summary = "Wait for a signal to be ready (becomes 1) and produce a token.";

  let description = [{
    The operator creates a wait node that represents a spin lock waiting
    for a list of signals pointed by the `barrierPtr` to be set ready (set to 1).

    Example:

    ```mlir
    token = tle.dsa.ascend.wait() {barrierPtr, num_barriers, scope = 1 : i32, semantic = 0 : i32}
    ```
  }];
  let arguments = (
    ins
    TT_PtrLike:$barrierPtr,
    TT_IntLike:$numBarriers,
    TT_Int: $waitValue,
    DefaultValuedAttr<TT_MemSyncScopeAttr, "MemSyncScope::GPU">:$scope,
    DefaultValuedAttr<TT_MemSemanticAttr, "MemSemantic::ACQUIRE">:$semantic
  );

  let results = (outs TT_IntLike:$token);

  let assemblyFormat = [{
      $barrierPtr `,` $numBarriers `,` $waitValue `,`
      `scope` `=` $scope `semantic` `=` $semantic
      attr-dict `:` type($barrierPtr) `,` type($numBarriers) `,` type($waitValue) `->` type($token)
    }];
}

def TLE_ConsumeTokenOp : TLE_Op<"consume_token", [
  Elementwise,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  TypesMatchWith<"result matches input type", "input", "result", "$_self">,
]> {
  let summary = "Consumes a token and builds data dependency between wait ops and input";

  let description = [{
    The operator creates a consume node that represents a data dependency between a wait node
    that produces a token and another node that uses the input data (e.g., load).

    Example:

    ```mlir
    tle.dsa.ascend.consume_token() {input, token}
    ```
  }];
  let arguments = (
    ins
    AnyTypeOf<[TT_Type, TT_TensorDescType]>:$input,
    TT_IntLike:$token
  );

  let results = (outs AnyTypeOf<[TT_Type, TT_TensorDescType]>:$result);

  let assemblyFormat = [{
      $input `,` $token `,`
      attr-dict `:` type($input) `,` type($token) `->` type($result)
    }];
}

def TLE_NotifyOp : TLE_Op<"notify", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
]> {
  let summary = "Update sigAddr with signalVal using operation sigOp on the specified rank";

  let arguments = (ins
    TT_Ptr:$sigAddr,
    I64:$signalVal,
    I32:$rank,
    TT_SignalOpAttr:$sigOp,
    TT_CommScopeAttr:$commScope
  );

}


#endif
