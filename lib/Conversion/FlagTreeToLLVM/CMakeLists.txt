add_triton_library(FlagTreeToLLVM
    DSLRegionOpToLLVM.cpp
    ExtractOpToLLVM.cpp
    PackOpToLLVM.cpp

    DEPENDS
    TritonGPUTypeInterfacesIncGen
)
