add_triton_library(TritonHCUGPUDialectToLLVM
    TritonHCUGPUToLLVMPatterns.cpp
    ExtractSliceOpToLLVM.cpp
    InThreadTransposeOpToTTG.cpp
    ConcatOpToLLVM.cpp
    ScaledUpcastToLLVM.cpp

    DEPENDS
    TritonHCUGPUIR
)
