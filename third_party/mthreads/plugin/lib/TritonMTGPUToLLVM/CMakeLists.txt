add_triton_library(TritonMTGPUToLLVM
    ConvertLayoutOpToLLVM.cpp
    DotOpToLLVM/WMMA.cpp
    DotOpToLLVM/SQMMA.cpp
    DotOpToLLVM/MTFMA.cpp
    DotOpToLLVM.cpp
    ElementwiseOpToLLVM.cpp
    LoadStoreOpToLLVM.cpp
    BarrierOpToLLVM.cpp
    TritonGPUToLLVM.cpp
    SPMDOpToLLVM.cpp
    Utility.cpp
    TargetInfo.cpp
    MUSATranslation.cpp
    FuncOpToLLVM.cpp
    BuiltinFuncToLLVM.cpp
    ReduceOpToLLVM.cpp
    AccelerateMTGPUMatmul.cpp
    MarkInplaceLoads.cpp
    Passes.cpp
    InplaceLoadToLLVM.cpp

    DEPENDS
    TritonMTGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    MLIRUBToLLVM
    MLIRGPUToMTGPUTransforms
)
