if(FLAGTREE_ILUVATAR_TLE)
    set(_TLE_IR_LIBS IluvatarTleIR)
elseif(FLAGTREE_TLE)
    set(_TLE_IR_LIBS TleIR) # flagtree tle raw
else()
    set(_TLE_IR_LIBS "")
endif()

if(FLAGTREE_BACKEND STREQUAL "iluvatar" AND NOT TRITON_BUILD_PROTON)
    set(_PROTON_IR_LIBS "")
else()
    set(_PROTON_IR_LIBS ProtonIR)
endif()

add_triton_library(TritonToTritonGPU
    RelayoutTritonGPU.cpp
    TritonGPUConversion.cpp
    TritonToTritonGPUPass.cpp

    DEPENDS
    TritonConversionPassIncGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUCTAAttrIncGen
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    ${_PROTON_IR_LIBS}
    TritonGPUIR
    ${_TLE_IR_LIBS}
)
