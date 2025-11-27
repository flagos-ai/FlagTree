get_flagtree_backend_lib("TritonToTritonGPU" _EXTRA_LINK_LIBS)

add_triton_library(TritonToTritonGPU
    TritonGPUConversion.cpp
    TritonToTritonGPUPass.cpp

    DEPENDS
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    ${_EXTRA_LINK_LIBS}
)
