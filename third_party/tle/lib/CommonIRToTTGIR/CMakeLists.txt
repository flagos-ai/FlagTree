add_triton_library(TleCommonIRToTTGIR
  CommonIRToTTGIR.cpp

  DEPENDS
  TleCommonIRToTTGIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTransformUtils
  CommonIRIR
  TritonIR
  TritonGPUIR
)

if(FLAGTREE_BACKEND STREQUAL "mthreads")
  target_compile_definitions(TleCommonIRToTTGIR PRIVATE FLAGTREE_COMMON_IR_MTHREADS)
endif()
