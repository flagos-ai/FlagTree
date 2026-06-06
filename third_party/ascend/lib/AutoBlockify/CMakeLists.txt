add_triton_library(AutoBlockify
        AutoBlockify.cpp
        RewriteOperation.cpp
        Utils.cpp

  DEPENDS
  AutoBlockifyPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  TritonIR
  TritonTransforms
  TritonAnalysis
  MLIRTransforms
  MLIRSupport
  MLIRSCFTransforms
)