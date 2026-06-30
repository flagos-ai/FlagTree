set(TLE_TO_LLVM_SRC
    DistributedBarrierOpToLLVM.cpp
    DSLRegionOpToLLVM.cpp
    ExclusiveCumsumOpToLLVM.cpp
    ExtractOpToLLVM.cpp
    LocalPointersOpToLLVM.cpp
    PackOpToLLVM.cpp
    RemotePointerUtils.cpp
    TMAStoreCommitGroupOpToLLVM.cpp
    ViewOpToLLVM.cpp
    WGMMAFenceOpToLLVM.cpp
)

if(FLAGCX_ENABLED)
 list(APPEND TLE_TO_LLVM_SRC
  GetDeviceIdToFlagCX.cpp
  GetLocalRankOpToLLVM.cpp
)
endif()

add_triton_library(TleToLLVM
    ${TLE_TO_LLVM_SRC}

    DEPENDS
    TleTableGen
    TritonGPUTypeInterfacesIncGen
    TleTools

    LINK_LIBS PUBLIC
    TleTools
)
