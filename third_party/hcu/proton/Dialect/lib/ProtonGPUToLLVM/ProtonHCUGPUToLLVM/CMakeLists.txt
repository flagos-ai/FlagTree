include_directories(${PROJECT_SOURCE_DIR}/third_party/hcu/include)

add_triton_library(ProtonHCUGPUToLLVM
    TargetInfo.cpp
    HCUPatternProtonGPUOpToLLVM.cpp
    AddSchedBarriers.cpp
    ConvertProtonGPUToLLVM.cpp

    DEPENDS
    ProtonHCUGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonGPUToLLVM
    TritonHCUGPUToLLVM
)
