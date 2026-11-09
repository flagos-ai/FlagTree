add_mlir_library(AscendModelAnalysis
  PARTIAL_SOURCES_INTENDED
  HardwareConfig.cpp
  HIVMAnalysis.cpp
  PipelineAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/AscendModel/Analysis

  DEPENDS
  AscendModelOpsIncGen

  LINK_LIBS PUBLIC
  AscendModelIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  LLVMSupport
)

if(TRITONSIM_HAS_BISHENGIR_HIVM)
  target_include_directories(AscendModelAnalysis SYSTEM PUBLIC
    ${TRITONSIM_BISHENGIR_INCLUDE_DIRS}
  )
  target_link_libraries(AscendModelAnalysis PUBLIC
    ${TRITONSIM_BISHENGIR_HIVM_LIBS}
    MLIRBufferizationDialect
    MLIRDLTIDialect
    MLIRDestinationStyleOpInterface
    MLIRLinalgDialect
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRSideEffectInterfaces
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRViewLikeInterface
  )
endif()
