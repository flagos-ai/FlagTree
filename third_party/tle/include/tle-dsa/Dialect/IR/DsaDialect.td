//===- DsaDialect.td - TLE DSA dialect --------------------*- tablegen -*-===//
//
// Template dialect for TLE-Struct style DSA extensions.
//
//===----------------------------------------------------------------------===//

#ifndef TLE_DSA_DIALECT
#define TLE_DSA_DIALECT

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition.
//===----------------------------------------------------------------------===//

def Dsa_Dialect : Dialect {
  let name = "dsa";
  let summary = "TLE DSA struct dialect (template)";
  let cppNamespace = "::mlir::dsa";
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Type definitions.
//===----------------------------------------------------------------------===//

class Dsa_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Dsa_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def DsaBufferType : Dsa_Type<"Buffer", "buffer"> {
  let summary = "Opaque local buffer handle";
  let description = [{
    This is a minimal, parameterized buffer handle type for DSA extensions.
    The backend defines the lowering semantics (e.g. mapping to SRAM/UB/NZ, etc).
  }];

  let parameters = (ins
    "Type":$elementType,
    OptionalParameter<"Attribute">:$memorySpace
  );

  let builders = [
    TypeBuilder<(ins
      "Type":$elementType,
      CArg<"Attribute", "nullptr">:$memorySpace), [{
        return Base::get($_ctxt, elementType, memorySpace);
      }]>
  ];
  let skipDefaultBuilders = 1;

  let assemblyFormat = "`<` $elementType (`,` $memorySpace^)? `>`";
}

//===----------------------------------------------------------------------===//
// Base op definition.
//===----------------------------------------------------------------------===//

class Dsa_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dsa_Dialect, mnemonic, traits>;

#endif // TLE_DSA_DIALECT
