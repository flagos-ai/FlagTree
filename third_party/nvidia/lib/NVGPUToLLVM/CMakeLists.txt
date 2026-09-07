add_triton_library(NVGPUToLLVM
    NVGPUToLLVMPass.cpp

    DEPENDS
    NVGPUConversionPassIncGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUCTAAttrIncGen
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen

    LINK_LIBS PUBLIC
    NVGPUIR
    TritonGPUIR
)
