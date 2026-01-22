add_triton_library(FlagTree_sunrise_TritonToTritonGPU
    RelayoutTritonGPU.cpp
    TritonGPUConversion.cpp
    TritonToTritonGPUPass.cpp

    DEPENDS
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    ProtonIR
    TritonGPUIR
)
