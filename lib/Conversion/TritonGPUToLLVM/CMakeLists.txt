if (FLAGTREE_BACKEND)
    set(NVGPUIR "")
    set(NVGPUTransforms "")
else()
    set(NVGPUIR "NVGPUIR")
    set(NVGPUTransforms "TritonNvidiaGPUTransforms")
endif()

get_flagtree_backend_lib("TritonGPUToLLVM" _EXTRA_LINK_LIBS)

add_triton_library(TritonGPUToLLVM
    ConvertLayoutOpToLLVM/SharedToDotOperandFMA.cpp
    DotOpToLLVM/FMA.cpp
    TypeConverter.cpp
    Utility.cpp
    ElementwiseOpToLLVM.cpp
    MemoryOpToLLVM.cpp
    AssertOpToLLVM.cpp
    ViewOpToLLVM.cpp
    MakeRangeOpToLLVM.cpp
    HistogramOpToLLVM.cpp
    AllocateSharedMemory.cpp
    ReduceOpToLLVM.cpp
    ScanOpToLLVM.cpp
    ConvertLayoutOpToLLVM.cpp
    ControlFlowOpToLLVM.cpp
    FuncOpToLLVM.cpp
    SPMDOpToLLVM.cpp
    DecomposeUnsupportedConversions.cpp
    PrintOpToLLVM.cpp

    DEPENDS
    TritonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRGPUDialect
    MLIRGPUToNVVMTransforms
    MLIRGPUToROCDLTransforms
    MLIRGPUTransforms
    TritonAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    ${NVGPUTransforms}
    ${NVGPUIR}
    ${_EXTRA_LINK_LIBS}
)
