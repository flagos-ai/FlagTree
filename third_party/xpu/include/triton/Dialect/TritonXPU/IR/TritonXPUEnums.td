#ifndef TRITONXPU_ENUMS
#define TRITONXPU_ENUMS

include "mlir/IR/EnumAttr.td"

def TT_DtypeAttr : I32EnumAttr<
    "Dtype", "",
    [
        I32EnumAttrCase<"UNKNOWN", 0, "unknown">,
        I32EnumAttrCase<"FP32", 1, "fp32">,
        I32EnumAttrCase<"FP16", 2, "fp16">,
    ]> {
    let cppNamespace = "::mlir::triton";
}

def TT_OffsetStatePolicyAttr : I32EnumAttr<
    "OffsetStatePolicy", "",
    [
        I32EnumAttrCase<"EMPTY", 0, "empty">,
        I32EnumAttrCase<"UNKNOWN", 1, "unknown">,
        I32EnumAttrCase<"DISCRETE_SAME", 2, "discrete_same">,
        I32EnumAttrCase<"CONTINUOUS", 3, "continuous">,
        I32EnumAttrCase<"DISCRETE", 4, "discrete">,
        I32EnumAttrCase<"LOCALLY_CONTINUOUS", 5, "locally_continuous">
    ]> {
    let cppNamespace = "::mlir::triton";
}

def TT_SyncAttr : I32EnumAttr<
    "MemorySyncMode", "",
    [
        I32EnumAttrCase<"SYNC", 1, "sync">,
        I32EnumAttrCase<"ASYNC", 2, "async">,
    ]> {
    let cppNamespace = "::mlir::triton";
}

#endif // TRITONXPU_ENUMS
