if(BUILD_MCTLE)
    set(_MCTLE_IR_LIBS McTleIR)
else()
    set(_MCTLE_IR_LIBS "")
endif()

add_triton_library(TritonToTritonGPU
    RelayoutTritonGPU.cpp
    ${PROJECT_SOURCE_DIR}/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp
    ${PROJECT_SOURCE_DIR}/lib/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.cpp

    DEPENDS
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    ProtonIR
    TritonGPUIR
    ${_MCTLE_IR_LIBS}
)

if(BUILD_MCTLE)
    target_include_directories(TritonToTritonGPU PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/metax/plugin
    ${PROJECT_BINARY_DIR}/third_party/metax/plugin
    )
endif()
