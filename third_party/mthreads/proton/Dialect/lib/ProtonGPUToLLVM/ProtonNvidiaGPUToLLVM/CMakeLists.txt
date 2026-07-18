# include_directories(${PROJECT_SOURCE_DIR}/third_party/nvidia/include)

# add_triton_library(ProtonNVIDIAGPUToLLVM
#     TargetInfo.cpp
#     NvidiaPatternProtonGPUOpToLLVM.cpp
#     ConvertProtonGPUToLLVM.cpp

#     DEPENDS
#     ProtonNvidiaGPUConversionPassIncGen

#     LINK_LIBS PUBLIC
#     ProtonGPUToLLVM
#     TritonNVIDIAGPUToLLVM
# )

# target_include_directories(ProtonNVIDIAGPUToLLVM PRIVATE
#   ${PROJECT_SOURCE_DIR}/third_party/mthreads/proton/Dialect/include
# )

add_triton_library(ProtonNVIDIAGPUToLLVM
    TargetInfo.cpp
    NvidiaPatternProtonGPUOpToLLVM.cpp
    ConvertProtonGPUToLLVM.cpp
    ../../compat/TargetInfo.cpp
    ../../compat/PTXAsmFormat.cpp
    ../../compat/Utility.cpp

    DEPENDS
    ProtonNvidiaGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonGPUToLLVM
    # TritonNVIDIAGPUToLLVM
)

target_include_directories(ProtonNVIDIAGPUToLLVM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/mthreads/proton/Dialect/include
)
