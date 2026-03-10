add_triton_library(TleToLLVM
    DSLRegionOpToLLVM.cpp
    ExtractOpToLLVM.cpp
    LocalPointersOpToLLVM.cpp
    PackOpToLLVM.cpp

    DEPENDS
    TleTableGen
    TritonGPUTypeInterfacesIncGen
)
