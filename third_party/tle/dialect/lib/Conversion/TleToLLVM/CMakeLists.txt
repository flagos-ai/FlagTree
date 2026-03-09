add_triton_library(TleToLLVM
    DistributedBarrierOpToLLVM.cpp
    DSLRegionOpToLLVM.cpp
    ExtractOpToLLVM.cpp
    LocalPointersOpToLLVM.cpp
    PackOpToLLVM.cpp
    RemotePointerUtils.cpp

    DEPENDS
    TritonGPUTypeInterfacesIncGen
)
