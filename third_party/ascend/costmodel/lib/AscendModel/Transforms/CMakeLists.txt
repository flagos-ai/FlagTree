add_mlir_library(AscendModelTransforms
  ConvertTritonToAscend.cpp
  InsertDataTransfers.cpp
  AssignOpIDs.cpp
  EstimateCycles.cpp
  HIVMAnalysisPass.cpp
  PipelineAnalysisPass.cpp
  PerfReportPass.cpp
  PassRegistration.cpp

  DEPENDS
  AscendModelOpsIncGen
  AscendModelTransformsPassIncGen

  LINK_LIBS PUBLIC
  AscendModelIR
  AscendModelAnalysis
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRSCFDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRTensorDialect
)

# Conditionally link Triton dialect library and add dependency
if(TRITONSIM_HAS_TRITON AND TRITON_DIALECT_LIB)
  add_dependencies(obj.AscendModelTransforms TritonDialectIncGen)
  target_link_libraries(obj.AscendModelTransforms PRIVATE ${TRITON_DIALECT_LIB})
endif()
