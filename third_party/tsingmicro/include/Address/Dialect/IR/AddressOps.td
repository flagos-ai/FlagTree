//===- AddressOps.td - Address dialect ops -----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ADDRESS_OPS
#define ADDRESS_OPS

include "Address/Dialect/IR/AddressDialect.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def Address_ConstantOp : Address_Op<"constant", [
    ConstantLike, Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Creates an address constant.";
  let description = [{
    The `addr.constant` operation produces an address-typed SSA value equal to
    some index constant.

    Example:

    ```mlir
    %addr0 = address.constant 0
    %addr1 = address.constant 1 : !address<3 : i32>
    ```
  }];
  let arguments = (ins IndexAttr:$value);
  let results = (outs AddressType:$result);
  let builders = [
    OpBuilder<(ins "int64_t":$value, CArg<"Attribute", "nullptr">:$addressSpace)>
  ];
  let assemblyFormat = "attr-dict $value custom<AddressType>(type($result))";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// TypeOffsetOp
//===----------------------------------------------------------------------===//

def Address_TypeOffsetOp : Address_Op<"type_offset", [ConstantLike, Pure]> {
  let summary = "Creates a type offset constant.";
  let description = [{
    The `addr.type_offset` operation produces an int or index-typed SSA value
    equal to a target-specific constant representing the offset of a single
    element of the given type. The default return type is `index`.
    Example:

    ```mlir
    %0 = addr.type_offset f32
    %1 = addr.type_offset memref<12 x f64> : i32
    ```
  }];

  let arguments = (ins TypeAttr:$baseType);
  let results = (outs AnySignlessIntegerOrIndex:$result);
  let builders = [
    OpBuilder<(ins "TypeAttr":$baseType, CArg<"Type", "nullptr">:$resultTy)>
  ];
  let assemblyFormat = [{
     attr-dict $baseType custom<IntType>(type($result))
  }];
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def Address_CastOp : Address_Op<"cast", [Pure]> {
  let summary = "Creates an address space cast";
  let description = [{
    The `addr.cast` operation casts addresses between address spaces.
    Example:

    ```mlir
    %addr = addr.cast %addr : !address to !address<1 : i32>
    ```
  }];
  let arguments = (ins AddressType:$input);
  let results = (outs AddressType:$result);
  let builders = [
    OpBuilder<(ins "Attribute":$addressSpace, "Value":$input)>
  ];
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// CastIntOp
//===----------------------------------------------------------------------===//

def Address_CastIntOp : Address_Op<"cast_int", [
   Pure, DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Creates an int <-> address cast.";
  let description = [{
    The `addr.cast_int` operation casts an int or index value to an address and
    vice-versa.
    Example:

    ```mlir
    %addr = addr.cast_int %int : i32 to !address<1 : i32>
    %index = addr.cast_int %addr : !address<1 : i32> to index
    ```
  }];
  let arguments = (ins AnyTypeOf<[AnySignlessIntegerOrIndex, AddressType]>:$input);
  let results = (outs AnyTypeOf<[AnySignlessIntegerOrIndex, AddressType]>:$result);
  let builders = [
    OpBuilder<(ins "Value":$input, CArg<"Type", "nullptr">:$resultTy)>
  ];
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($result)";
}

//===----------------------------------------------------------------------===//
// FromMemrefOp
//===----------------------------------------------------------------------===//

def Address_FromMemRefOp : Address_Op<"from_memref", [Pure]> {
  let summary = "Converts a memref to an address.";
  let description = [{
    The `addr.from_memref` operation extracts the aligned or allocated address
    from a memref.
    Example:

    ```mlir
    %addr = addr.from_memref [%memref : memref<2x4x?xf32>]
    %baseAddr = addr.from_memref extract_base [%memref : memref<1xf32, 3>]
    ```
  }];
  let arguments = (ins AnyMemRef:$input, UnitAttr:$extract_base);
  let results = (outs AddressType:$result);
  let assemblyFormat = [{
    (`extract_base` $extract_base^)? ` ` `[` $input
    custom<FromMemRef>(type($input), type($result)) `]` attr-dict
  }];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def Address_FromUnrankedMemRefOp : Address_Op<"from_unranked_memref", [Pure]> {
  let summary = "Converts a unranked memref to an address.";
  let description = [{
    The `addr.from_unranked_memref` operation extracts the aligned or allocated address
    from a memref.
    Example:

    ```mlir
    %addr = addr.from_unranked_memref [%memref : memref<*xf32>]
    %baseAddr = addr.from_unranked_memref extract_base [%memref : memref<*xf32, 3>]
    ```
  }];
  let arguments = (ins AnyUnrankedMemRef:$input, UnitAttr:$extract_base);
  let results = (outs AddressType:$result);
  let assemblyFormat = [{
    (`extract_base` $extract_base^)? ` ` `[` $input
    custom<FromUnrankedMemRef>(type($input), type($result)) `]` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ToMemRefOp
//===----------------------------------------------------------------------===//

def Address_ToMemRefOp : Address_Op<"to_memref", [Pure]> {
  let summary = "Converts an address to a memref.";
  let description = [{
    The `addr.to_memref` operation converts an `address` to a `memref` with
    static shape.
    Example:

    ```mlir
    %memref = addr.to_memref %addr : memref<2x4xf32>
    %memrefFull = addr.to_memref %addr base %baseAddr : memref<2xf32>
    ```
  }];
  let arguments = (ins AddressType:$address, Optional<AddressType>:$base);
  let results = (outs AnyStaticShapeMemRef:$result);
  let assemblyFormat = [{
    $address attr-dict
    custom<ToMemRef>($base, type($base), type($address), type($result))
  }];
  let builders = [
    OpBuilder<(ins "MemRefType":$type, "Value":$address)>
  ];
  let hasVerifier = 1;
  let hasCanonicalizeMethod = 1;
}

def Address_ToUnrankedMemRefOp : Address_Op<"to_unranked_memref", [Pure]> {
  let summary = "Converts an address to an unranked memref.";
  let description = [{
    The `addr.to_unranked_memref` operation converts an `address` to an unranked `memref`.
    Example:

    ```mlir
    %memref = addr.to_unranked_memref %addr : memref<*xf32>
    %memrefFull = addr.to_unranked_memref %addr base %baseAddr : memref<*xf32>
    ```
  }];
  let arguments = (ins AddressType:$address, Optional<AddressType>:$base);
  let results = (outs AnyUnrankedMemRef:$result);
  let assemblyFormat = [{
    $address attr-dict
    custom<ToUnrankedMemRef>($base, type($base), type($address), type($result))
  }];
  let builders = [
    OpBuilder<(ins "UnrankedMemRefType":$type, "Value":$address)>
  ];
}

//===----------------------------------------------------------------------===//
// PtrAddOp
//===----------------------------------------------------------------------===//

def Address_PtrAddOp : Address_Op<"ptradd", [
    Pure, AllTypesMatch<["base", "result"]>
  ]> {
  let summary = "Adds an int or index to a addres.";
  let description = [{
    The `addr.ptradd` operation adds an `address` and an integer or index to
    produce a new address.
    Example:
    ```mlir
    %addr = addr.ptradd %addr : !addr.address<3 : i32>, %c10 : i32
    ```
  }];

  let arguments = (ins AddressType:$base, AnySignlessIntegerOrIndex:$offset);
  let results = (outs AddressType:$result);

  let assemblyFormat = [{
    $base custom<AddressType>(type($base)) `,` $offset
    custom<IntType>(type($offset)) attr-dict
  }];
}

#endif // ADDRESS_OPS
