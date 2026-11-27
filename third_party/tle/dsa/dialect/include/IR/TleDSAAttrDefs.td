// Copyright 2026- Xcoresigma Technology Co., Ltd

#ifndef TRITON_TLE_DSA_ATTR_DEFS
#define TRITON_TLE_DSA_ATTR_DEFS

include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "tle/dsa/dialect/include/IR/TleDSADialect.td"

// Signal operation type used by tle notify.
def TT_SignalOpAttr : I32EnumAttr<
    "SignalOp", "",
    [
        I32EnumAttrCase<"SET", 1, "set">,
        I32EnumAttrCase<"ADD", 2, "add">,
    ]> {
    let cppNamespace = "::mlir::triton::distributed";
}

// Communication scope used by tle notify.
def TT_CommScopeAttr : I32EnumAttr<
    "CommScope", "",
    [
      I32EnumAttrCase<"GPU", 1, "gpu">,
      I32EnumAttrCase<"INTRA_NODE", 2, "intra_node">,
      I32EnumAttrCase<"INTER_NODE", 3, "inter_node">,
    ]> {
    let cppNamespace = "::mlir::triton::distributed";
}

#endif
