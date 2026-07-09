if(FLAGTREE_TLE)
    set(_TLE_IR_LIBS TleIR) # flagtree tle raw
else()
    set(_TLE_IR_LIBS "")
endif()

get_flagtree_backend_lib("TritonToTritonGPU" _EXTRA_LINK_LIBS)

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
    ProtonIR
    TritonGPUIR
    ${_TLE_IR_LIBS}
    ${_EXTRA_LINK_LIBS}
)
