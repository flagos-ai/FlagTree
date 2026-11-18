add_triton_library(TritonControlFlowOpt
        BlockPtrDecompose.cpp
        CFGStructuring.cpp
        ControlFlowAnalysis.cpp
        ControlFlowRewrite.cpp
        TensorPtrDecompose.cpp
        TritonControlFlowOptPass.cpp

  DEPENDS
  TritonControlFlowOptPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRScopeDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRSupport
  MLIRTritonNPUUtils
  TritonIR
)
