add_triton_library(TritonDistributedToLLVM
    HCU/LibDeviceToLLVM.cpp
    HCU/DistributedOpToLLVM.cpp
    HCU/ConvertHCUDistributedToLLVM.cpp

    DEPENDS
    TritonDistributedToLLVMConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    DistributedIR
    TritonHCUGPUIR
    TritonHCUGPUToLLVM
    TritonHCUGPUDialectToLLVM
    TritonHCUGPUTransforms
    TritonHCUUtils
    TritonHCUAnalysis
)
