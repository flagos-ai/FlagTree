add_triton_library(TritonDistributedToTritonGPU
    TritonDistributedToTritonGPU.cpp

    DEPENDS
    TritonDistributedConversionPassIncGen
    TritonConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    TritonIR
    ProtonIR
    DistributedIR
    SIMTIR
    TritonGPUIR
    TritonGPUTransforms
)

if(TRITON_BUILD_FLAGPRISM)
  target_include_directories(TritonDistributedToTritonGPU PRIVATE
    "${FLAGPRISM_SOURCE_DIR}/Profiler/Dialect/include"
    "${FLAGPRISM_SOURCE_DIR}"
    "${PROJECT_BINARY_DIR}/third_party/FlagPrism")
endif()
