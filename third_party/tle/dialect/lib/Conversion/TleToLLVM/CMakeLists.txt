add_triton_library(TleToLLVM
    ExtractOpToLLVM.cpp
    LocalPointersOpToLLVM.cpp
    PackOpToLLVM.cpp

    DEPENDS
    TritonGPUTypeInterfacesIncGen
)
