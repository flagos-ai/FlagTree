include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_mlir_conversion_library(MLIRTritonGCUTransforms_${arch}
  GCUSupportVerifier.cpp
  Combine.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/triton_gcu/Transforms

  DEPENDS
  TritonGCUTableGen_${arch}
  TritonGCUTransformsPassIncGen_${arch}
  TritonGCUCombineIncGen_${arch}
  triton_${arch}

  LINK_LIBS PUBLIC
#  TritonIR
  MLIRGPUTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRSideEffectInterfaces
  )
set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonGCUCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonGCUCombineIncGen_${arch})
