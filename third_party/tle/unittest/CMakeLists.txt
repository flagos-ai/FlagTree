add_triton_ut(
  NAME TestTleCoalescingSlice
  SRCS CoalescingSliceTest.cpp
  LIBS
    MLIRAnalysis
    MLIRArithDialect
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRParser
)
