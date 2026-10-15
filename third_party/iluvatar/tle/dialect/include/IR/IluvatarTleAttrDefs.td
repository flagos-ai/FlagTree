#ifndef ILUVATAR_TLE_ATTR_DEFS
#define ILUVATAR_TLE_ATTR_DEFS

#ifdef __ILUVATAR_TLE__
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "IluvatarTleDialect.td"

// Mirrors tle/dialect/include/IR/TleAttrDefs.td: same cases, same values, same
// spellings. The shared TLE frontend converts to these before calling the
// builder, so the two dialects have to agree on them. Trunk's signal enums
// (SignalOpKind, SignalWaitKind) are absent because iluvatar implements no
// signal op, so binding them would be dead code.
//
// The values additionally match flagcxCoopKind_t / flagcxTeamKind_t /
// flagcxDeviceMemoryOrder_t, which is what lets the barrier lowering forward
// the enums straight to the FlagCX ABI.
def IluvatarTle_FlagCXCoopKindAttr : I32EnumAttr<
  "FlagCXCoopKind", "",
  [
    I32EnumAttrCase<"THREAD", 0, "thread">,
    I32EnumAttrCase<"WARP", 1, "warp">,
    I32EnumAttrCase<"BLOCK", 2, "block">,
  ]> {
  let cppNamespace = "::mlir::triton::iluvatar_tle";
}


def IluvatarTle_FlagCXTeamKindAttr : I32EnumAttr<
  "FlagCXTeamKind", "",
  [
    I32EnumAttrCase<"INTRA", 0, "intra">,
    I32EnumAttrCase<"INTER", 1, "inter">,
    I32EnumAttrCase<"WORLD", 2, "world">,
  ]> {
  let cppNamespace = "::mlir::triton::iluvatar_tle";
}


def IluvatarTle_SyncScopeAttr : I32EnumAttr<
  "SyncScope", "",
  [
    I32EnumAttrCase<"SYSTEM", 0, "system">,
    I32EnumAttrCase<"DEVICE", 1, "device">,
    I32EnumAttrCase<"BLOCK", 2, "block">,
    I32EnumAttrCase<"THREAD", 3, "thread">,
  ]> {
  let cppNamespace = "::mlir::triton::iluvatar_tle";
}


def IluvatarTle_MemoryOrderAttr : I32EnumAttr<"MemoryOrder", "",
  [
    I32EnumAttrCase<"RELAXED", 0, "relaxed">,
    I32EnumAttrCase<"ACQUIRE", 1, "acquire">,
    I32EnumAttrCase<"RELEASE", 2, "release">,
    I32EnumAttrCase<"ACQ_REL", 3, "acq_rel">,
  ]> {
  let cppNamespace = "::mlir::triton::iluvatar_tle";
}
#endif // __ILUVATAR_TLE__

#endif // ILUVATAR_TLE_ATTR_DEFS
