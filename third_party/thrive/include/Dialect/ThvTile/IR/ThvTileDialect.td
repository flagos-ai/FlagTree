#ifndef THVTILE_DIALECT
#define THVTILE_DIALECT

include "mlir/IR/OpBase.td"

def ThvTile_Dialect : Dialect {
  let name = "thvtile";
  let cppNamespace = "::mlir::thvtile";

  let description = [{
    ThvTile Dialect provides tile-level abstract operations for the Thv backend.

    It separates hardware-independent tile semantics from PE-level hardware
    details, enabling layered optimization.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect",
    "math::MathDialect",
    "func::FuncDialect"
  ];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
