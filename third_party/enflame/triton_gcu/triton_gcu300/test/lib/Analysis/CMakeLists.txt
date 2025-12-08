# Exclude tests from libMLIR.so
add_mlir_library(TritonGCUTestAnalysis_${arch}
  TestFirstLastUserAnalysis.cpp

  EXCLUDE_FROM_LIBMLIR

)

mlir_target_link_libraries(TritonGCUTestAnalysis_${arch} PUBLIC
  TritonGCUAnalysis_${arch}
  TritonGCUIR_${arch}
  MLIRPass
  MLIRAnalysis
  MLIRIR
)

target_include_directories(TritonGCUTestAnalysis_${arch}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../include
  ${CMAKE_CURRENT_BINARY_DIR}/../../../include
  )
