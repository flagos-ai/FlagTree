if(TRITON_BUILD_PROTON)
    set(_ILUVATAR_PROTON_LIBS ProtonIR)
else()
    set(_ILUVATAR_PROTON_LIBS "")
endif()

add_triton_library(TritonToTritonGPU
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
    ${_ILUVATAR_PROTON_LIBS}
    TritonGPUIR
)
