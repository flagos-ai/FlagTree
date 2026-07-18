#ifndef MUSA_ATTRDEFS
#define MUSA_ATTRDEFS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "MUSADialect.td"

class MUSA_Attr<string name, list<Trait> traits = [],
                string baseCppClass = "::mlir::Attribute">
  : AttrDef<MUSA_Dialect, name, traits, baseCppClass> {
}

def TME_SwizzleGranularityAttr : I32EnumAttr<"TMESwizzleGranularity",
    "TME swizzle granularity",
    [
      I32EnumAttrCase<"SG_NONE", 0>,
      I32EnumAttrCase<"SG_16B", 1>,
      I32EnumAttrCase<"SG_32B", 2>,
      I32EnumAttrCase<"SG_64B", 3>,
      I32EnumAttrCase<"SG_128B", 4>
    ]> {
  let cppNamespace = "::mlir::triton::musa";
}

def TME_SwizzleStrideAttr : I32EnumAttr<"TMESwizzleStride",
    "TME swizzle stride",
    [
      I32EnumAttrCase<"SS_32B", 0>,
      I32EnumAttrCase<"SS_64B", 1>,
      I32EnumAttrCase<"SS_128B", 2>,
      I32EnumAttrCase<"SS_256B", 3>
    ]> {
  let cppNamespace = "::mlir::triton::musa";
}

def TME_SwizzleLineAttr : I32EnumAttr<"TMESwizzleLine",
    "TME swizzle line",
    [
      I32EnumAttrCase<"SL_128B", 0>,
      I32EnumAttrCase<"SL_256B", 1>
    ]> {
  let cppNamespace = "::mlir::triton::musa";
}

def TME_PrefetchSizeAttr : I32EnumAttr<"TMEPrefetchSize",
    "TME prefetch size",
    [
      I32EnumAttrCase<"SZ_NONE", 0>,
      I32EnumAttrCase<"SZ_64B", 64>,
      I32EnumAttrCase<"SZ_128B", 128>,
      I32EnumAttrCase<"SZ_256B", 256>
    ]> {
  let cppNamespace = "::mlir::triton::musa";
}

def TME_PersistenceAttr : I32EnumAttr<"TMEPersistence",
    "TME persistence policy",
    [
      I32EnumAttrCase<"CACHE_NONE", 0>,
      I32EnumAttrCase<"CACHE_ONCE", 1>,
      I32EnumAttrCase<"CACHE_NORMAL", 2>,
      I32EnumAttrCase<"CACHE_PERSIST", 3>,
      I32EnumAttrCase<"NO_OVERRIDE", 4>,
      I32EnumAttrCase<"LAST_USE", 5>
    ]> {
  let cppNamespace = "::mlir::triton::musa";
}

def TME_CachePolicyAttr : I32EnumAttr<"TMEL2CachePolicy",
    "TME L2 cache policy",
    [
      I32EnumAttrCase<"NEW_ALLOC", 0>,
      I32EnumAttrCase<"BYPASS", 1>
    ]> {
  let cppNamespace = "::mlir::triton::musa";
}

#endif
