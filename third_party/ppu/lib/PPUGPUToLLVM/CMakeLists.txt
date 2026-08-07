add_triton_library(PPUGPUToLLVM
    PPUGPUToLLVMPass.cpp

    DEPENDS
    PPUGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    PPUGPUIR
)
