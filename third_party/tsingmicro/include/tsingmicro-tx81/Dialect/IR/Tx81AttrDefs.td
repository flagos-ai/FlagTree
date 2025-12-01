//===---------------------- Tx81AttrDefs.td -------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#ifndef TSINGMICRO_TX81_ATTR_DEFS
#define TSINGMICRO_TX81_ATTR_DEFS

include "mlir/IR/EnumAttr.td"

// Round mode, aligned with RND_MODE in instr_def.h
def RoundModeAttr : I32EnumAttr<"RoundMode", "Round mode", [
  I32EnumAttrCase<"RND_NEAREST_EVEN", 0, "nearest">,
  I32EnumAttrCase<"RND_ZERO", 1, "zero">,
  I32EnumAttrCase<"RND_POS_INF", 2, "pos">,
  I32EnumAttrCase<"RND_NEG_INF", 3, "neg">,
  I32EnumAttrCase<"RND_STOCHASTIC", 4, "stochastic">
]> {
  let cppNamespace = "::mlir::tx";
}

#endif // TSINGMICRO_TX81_ATTR_DEFS
