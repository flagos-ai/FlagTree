//===- AddressDialect.td - Address dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ADDRESS_DIALECT
#define ADDRESS_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Address dialect definition.
//===----------------------------------------------------------------------===//

def Address_Dialect : Dialect {
    let name = "addr";
    let summary = "address dialect";
    let cppNamespace = "::mlir::addr";
    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Address type definitions
//===----------------------------------------------------------------------===//

class Address_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Address_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def AddressType : Address_Type<"Address", "address", [
    MemRefElementTypeInterface
  ]> {
  let summary = "Type for holding addresses";
  let description = [{
    Syntax:

    ```mlir
    address ::= `address` `<` (address-space)? `>`
    address-space ::= attribute-value
    ```
    `address` is a type for representing memory addresses, including its address
    space. Its size is target and address space dependent, and it is only known
    once it is lowered.
  }];
  let parameters = (ins OptionalParameter<"Attribute">:$addressSpace);
  let builders = [
    TypeBuilder<(ins
      CArg<"Attribute", "{}">:$addressSpace), [{
        return Base::get($_ctxt, addressSpace);
      }]>,
    TypeBuilderWithInferredContext<(ins
      "MemRefType":$type), [{
        assert(type && "expected a valid memref type");
        return Base::get(type.getContext(), type.getMemorySpace());
      }]>
  ];
  let assemblyFormat = "(`<` $addressSpace^ `>`)?";
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// Base address operation definition.
//===----------------------------------------------------------------------===//

class Address_Op<string mnemonic, list<Trait> traits = []> :
        Op<Address_Dialect, mnemonic, traits>;

#endif // ADDRESS_DIALECT
