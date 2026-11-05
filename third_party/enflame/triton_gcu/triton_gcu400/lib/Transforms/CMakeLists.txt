include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRTritonGCUTransforms_${arch}
  Combine.cpp
  GCUSupportVerifier.cpp
  GCUWarpSpecialization.cpp
  GCUTleLowerAsyncLoad.cpp
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
  # GCUWarpSpecialization.cpp and WarpSpecialization/WSCodePartition.cpp, as
  # well as Transforms/Passes.h, include "Dialect/GCUWS/IR/Dialect.h" and need
  # the generated GCUWS headers (Dialect.h.inc, Ops.h.inc, Types.h.inc,
  # GCUWSAttrDefs.h.inc, GCUWSAttrEnums.h.inc, GCUWSOpInterfaces.h.inc)
  # to exist before compiling.
  GCUWSTableGen_${arch}
  GCUWSAttrDefsIncGen_${arch}
  triton_gcu400

  LINK_LIBS PUBLIC
  MLIRGPUTransforms
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRSideEffectInterfaces
  )


# Triton 3.8 配套的 MLIR 把 arith.addf/subf/mulf/divf 改成了
# Arith_FloatBinaryOpWithRoundingMode（多一个 roundingmode 可选属性），DRR pattern
# 必须显式写出每个操作数/属性，故 3.8 起改用 Combine38.td（多一个 `$_` 占位）。
if(TRITON_VERSION VERSION_GREATER_EQUAL "38")
  set(LLVM_TARGET_DEFINITIONS Combine38.td)
else()
  set(LLVM_TARGET_DEFINITIONS Combine.td)
endif()
mlir_tablegen(TritonGCUCombine.inc -gen-rewriters)
add_public_tablegen_target(TritonGCUCombineIncGen_${arch})
