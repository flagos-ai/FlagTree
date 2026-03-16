// Copyright 2026- Xcoresigma Technology Co., Ltd

#ifndef TRITON_TLE_OPS
#define TRITON_TLE_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "triton/Dialect/Triton/IR/TritonTypeInterfaces.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"
include "tle/dsa/dialect/include/IR/TleDialect.td"

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

#endif
