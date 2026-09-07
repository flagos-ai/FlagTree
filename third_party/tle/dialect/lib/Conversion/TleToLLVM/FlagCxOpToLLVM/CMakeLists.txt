set(FLAGCX_TO_LLVM_SRC
    DeviceIntraBarrierOpToLLVM.cpp
    GetLocalRankOpToLLVM.cpp
    FlagCxSignalOpToLLVM.cpp
    GetWorldRankOpToLLVM.cpp
    FlagCxOpToLLVM.cpp
)

add_triton_library(FlagCxToLLVM
    ${FLAGCX_TO_LLVM_SRC}

    DEPENDS
    TleTableGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    ${TLE_TRITON_GPU_ATTR_DEPENDS}
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen
    TleTools

    LINK_LIBS PUBLIC
    TleTools
)
