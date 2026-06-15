add_triton_library(TleToLLVM
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

    DEPENDS
    TleTableGen
    TritonGPUTypeInterfacesIncGen
)
