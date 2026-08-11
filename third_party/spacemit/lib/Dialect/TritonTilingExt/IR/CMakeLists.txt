add_triton_library(TritonTilingExtIR
  BufferizableOpInterfaceImpl.cpp
  CumSum.cpp
  TritonTilingExtDialect.cpp

  DEPENDS
  TritonTilingExtInterfacesIncGen
  TritonTilingExtOpsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRAffineAnalysis
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLinalgUtils
  )
