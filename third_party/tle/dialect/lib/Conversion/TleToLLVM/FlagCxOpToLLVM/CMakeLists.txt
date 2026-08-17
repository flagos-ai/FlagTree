set(FLAGCX_TO_LLVM_SRC
    DeviceIntraBarrierOpToLLVM.cpp
    GetLocalRankOpToLLVM.cpp
<<<<<<< HEAD
    FlagCxSignalOpToLLVM.cpp
=======
    GetWorldRankOpToLLVM.cpp
>>>>>>> origin/main
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
