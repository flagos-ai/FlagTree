set(LINK_IR_LIBS
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  GluonIR
  TritonNvidiaGPUIR
)

if (TRITON_BUILD_DISTRIBUTED)
  list(APPEND LINK_IR_LIBS DistributedIR)
endif()

add_triton_library(TritonAnalysis
  ${PROJECT_SOURCE_DIR}/lib/Analysis/AxisInfo.cpp
  ${PROJECT_SOURCE_DIR}/lib/Analysis/Allocation.cpp
  ${PROJECT_SOURCE_DIR}/lib/Analysis/Membar.cpp
  ${PROJECT_SOURCE_DIR}/lib/Analysis/Alias.cpp
  ${PROJECT_SOURCE_DIR}/lib/Analysis/Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen
  McTleTableGen

  LINK_LIBS PUBLIC
  ${LINK_IR_LIBS}
)
