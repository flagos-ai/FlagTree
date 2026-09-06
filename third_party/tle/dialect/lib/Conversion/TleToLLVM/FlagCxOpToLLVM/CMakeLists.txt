set(FLAGCX_TO_LLVM_SRC
    FlagCxBarrierOpToLLVM.cpp
    GetLocalRankOpToLLVM.cpp
    FlagCxSignalOpToLLVM.cpp
    GetWorldRankOpToLLVM.cpp
    FlagCxOpToLLVM.cpp
)

add_triton_library(FlagCxToLLVM
    ${FLAGCX_TO_LLVM_SRC}

    DEPENDS
    TleTableGen
    TritonGPUTypeInterfacesIncGen
    TleTools

    LINK_LIBS PUBLIC
    TleTools
)
