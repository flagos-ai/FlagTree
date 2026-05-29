if(TRITON_BUILD_PROTON)
  set(_ProtonIRLib ProtonIR)
endif()

if(FLAGTREE_TLE)
  set(_TleIRLib TleIR)  # flagtree tle raw
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
    ${_ProtonIRLib}
    TritonGPUIR
    ${_TleIRLib}
)
