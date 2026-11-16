add_triton_library(TritonHCUAnalysis
  RangeAnalysis.cpp
  AxisInfoExt.cpp
  HCUGPUAllocation.cpp

  DEPENDS
  TritonTableGen
  TritonHCUGPUTableGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
)

if(FLAGTREE_TLE)
  add_dependencies(TritonHCUAnalysis TleTableGen)
endif()
