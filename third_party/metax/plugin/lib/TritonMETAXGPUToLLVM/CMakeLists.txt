add_triton_library(TritonMETAXGPUToLLVM
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/MMAMACA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    HistogramOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    ScanOpToLLVM.cpp
    ReduceOpToLLVM.cpp
    TensorPtrOpsToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    ViewOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    Fp4ToFpScaledOpToLLVM.cpp
    BuiltinFuncToLLVMExt.cpp

    DEPENDS
    TritonMETAXGPUTransformsIncGen
    TritonMETAXGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
)
