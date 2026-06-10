//===- LinalgEnumsExt.td - EVAS Linalg enum attrs ---------*- tablegen -*-===//

#ifndef EVAS_LINALGEXT_ENUMS
#define EVAS_LINALGEXT_ENUMS

include "mlir/IR/EnumAttr.td"

def EVAS_LINALG_EvaMindCastRndModeAttr : I64EnumAttr<
    "EvaMindCastRndMode", "",
    [
      I64EnumAttrCase<"RNE", 0>,
      I64EnumAttrCase<"RTZ", 1>,
      I64EnumAttrCase<"RDN", 2>,
      I64EnumAttrCase<"RUP", 3>,
      I64EnumAttrCase<"RMM", 4>,
    ]> {
  let cppNamespace = "::mlir::linalg";
}

#endif // EVAS_LINALGEXT_ENUMS

