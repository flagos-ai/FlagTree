add_triton_library(TleToLLVM
    DSLRegionOpToLLVM.cpp
    ExtractOpToLLVM.cpp
    PackOpToLLVM.cpp

    DEPENDS
    TritonGPUTypeInterfacesIncGen
)
