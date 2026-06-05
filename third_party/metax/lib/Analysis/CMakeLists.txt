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
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  ${LINK_IR_LIBS}
)
