add_triton_library(NVGPUToLLVM
    NVGPUToLLVMPass.cpp

    DEPENDS
    NVGPUConversionPassIncGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUCGAAttrIncGen
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen

    LINK_LIBS PUBLIC
    NVGPUIR
    TritonGPUIR
)
