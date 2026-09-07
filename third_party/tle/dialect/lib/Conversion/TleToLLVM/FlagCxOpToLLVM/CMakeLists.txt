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
    TritonGPUCTAAttrIncGen
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen
    TleTools

    LINK_LIBS PUBLIC
    TleTools
)
