if(FLAGTREE_MTHREADS_TLE)
    set(_MUSATLE_DEPS MUSATLETableGen)
    set(_MUSATLE_LIBS MUSATLEIR)
else()
    set(_MUSATLE_DEPS "")
    set(_MUSATLE_LIBS "")
endif()

add_triton_library(TritonToTritonGPU
    RelayoutTritonGPU.cpp
    TritonGPUConversion.cpp
    TritonToTritonGPUPass.cpp

    DEPENDS
    TritonConversionPassIncGen
    ${_MUSATLE_DEPS}

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    ProtonIR
    TritonGPUIR
    ${_MUSATLE_LIBS}
)
