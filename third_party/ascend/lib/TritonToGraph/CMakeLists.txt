add_triton_library(TritonToGraph
  AliasAnalysis.cpp
  ControlFlowGraph.cpp
  ControlFlowGraphBuilder.cpp
  DataflowGraph.cpp
  MemorySsaBuilder.cpp
  Passes.cpp

  DEPENDS
  TritonToGraphPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRDialectUtils
  MLIRIR
  MLIRFuncDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  TritonIR
  TritonTransforms
)
