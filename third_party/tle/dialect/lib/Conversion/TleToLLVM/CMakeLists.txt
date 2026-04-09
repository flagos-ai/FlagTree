add_triton_library(TleToLLVM
    DistributedBarrierOpToLLVM.cpp
    DSLRegionOpToLLVM.cpp
    ExclusiveCumsumOpToLLVM.cpp
    ExtractOpToLLVM.cpp
    LocalPointersOpToLLVM.cpp
    PackOpToLLVM.cpp
    RemotePointerUtils.cpp

    DEPENDS
    TleTableGen
    TritonGPUTypeInterfacesIncGen
)
