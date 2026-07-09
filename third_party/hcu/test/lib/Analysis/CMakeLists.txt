add_mlir_library(TritonHCUGPUTestAnalysis
  TestHCURangeAnalysis.cpp
  TestHCUGPUMembar.cpp
  TestAxisInfo.cpp

  DEPENDS
  TritonTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRPass
)
