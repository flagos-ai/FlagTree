include_directories(${CMAKE_CURRENT_BINARY_DIR})
add_mlir_conversion_library(MLIRTritonGCUTransforms_${arch}
  Combine.cpp
  GCUSupportVerifier.cpp
  GCUWarpSpecialization.cpp
  WarpSpecialization/CodePartitionUtility.cpp
  WarpSpecialization/TaskIdPropagation.cpp
  WarpSpecialization/Utility.cpp
  WarpSpecialization/WSBuffer.cpp
  WarpSpecialization/WSCodePartition.cpp
  WarpSpecialization/WSLowerMem.cpp
  WarpSpecialization/WSSpecialize.cpp
  WarpSpecialization/WSTaskIdPropagate.cpp
  WarpSpecialization/WSTaskPartition.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/triton_gcu/Transforms

  DEPENDS
  TritonGCUTableGen_${arch}
  TritonGCUTransformsPassIncGen_${arch}
  TritonGCUCombineIncGen_${arch}
  triton_${arch}

  LINK_LIBS PUBLIC
  MLIRGPUTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRSideEffectInterfaces
  )

target_compile_options(obj.MLIRTritonGCUTransforms_${arch}
  PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-error=comment>
)

set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(TritonGCUCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonGCUCombineIncGen_${arch})
