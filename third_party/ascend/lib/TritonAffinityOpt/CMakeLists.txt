add_triton_library(TritonAffinityOpt
  DAGSSBuffer.cpp
  DAG.cpp
  DAGSync.cpp
  DAGScope.cpp

  DEPENDS
  TritonAffinityOptConversionPassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
  MLIRSCFDialect
)
