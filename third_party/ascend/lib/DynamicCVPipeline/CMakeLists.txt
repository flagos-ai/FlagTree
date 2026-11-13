file(GLOB STANDARDIZE_OP_PASS_SRC CONFIGURE_DEPENDS
  "StandardizeOp/*.cpp"
)
file(GLOB PLAN_COMPUTE_BLOCK_PASS_SRC CONFIGURE_DEPENDS
  "PlanComputeBlock/*.cpp"
)
file(GLOB COMMON_SRC CONFIGURE_DEPENDS
  "Common/*.cpp"
)
file(GLOB SPLIT_DATAFLOW_SRC_FILES CONFIGURE_DEPENDS
    "SplitDataflow/*.cpp"
)
file(GLOB SEPARATE_MEMORY_FROM_COMPUTE_SRC_FILES CONFIGURE_DEPENDS
  "SeparateMemoryFromCompute/*.cpp"
)

file(GLOB ALLOC_MULTI_CACHE_SRCS_FILES CONFIGURE_DEPENDS
  "AllocMultiCache/*.cpp"
)

file(GLOB ADD_CONTROLFLOW_CONDITION_SRC_FILES CONFIGURE_DEPENDS
    "AddControlFlowCondition/*.cpp"
)

file(GLOB Compute_BLOCK_OPT_SRC_FILES CONFIGURE_DEPENDS
    "ComputeBlockOpt/*.cpp"
    "ComputeBlockOpt/SplitIfByBlockId/*.cpp"
)

file(GLOB ANALYZE_DATAFLOW_SRC_FILES CONFIGURE_DEPENDS
    "AnalyzeDataFlow/*.cpp"
)

file(GLOB PRE_CHECK_AVAILABLE_SRC_FILES CONFIGURE_DEPENDS
    "PreCheckAvailable/*.cpp"
)

add_triton_library(DynamicCVPipeline
  AddDynamicCVPipeline.cpp
  PreCheckAvailable.cpp

  StandardizeOp.cpp
  ${STANDARDIZE_OP_PASS_SRC}

  PlanComputeBlock.cpp
  ComputeBlockOptPass.cpp
  AnalyzeDataFlow.cpp
  ${PLAN_COMPUTE_BLOCK_PASS_SRC}
  ${COMMON_SRC}
  ${Compute_BLOCK_OPT_SRC_FILES}
  ${ANALYZE_DATAFLOW_SRC_FILES}
  ${PRE_CHECK_AVAILABLE_SRC_FILES}
  SplitDataflow.cpp
  SeparateMemoryFromComputePass.cpp
  SeparateMemoryFromCompute/MarkGMLoadPass.cpp
  AllocMultiCache.cpp
  AddControlFlowCondition.cpp
  ${PLAN_COMPUTE_BLOCK_SRC_FILES}
  ${SPLIT_DATAFLOW_SRC_FILES}
  ${SEPARATE_MEMORY_FROM_COMPUTE_SRC_FILES}
  ${ALLOC_MULTI_CACHE_SRCS_FILES}
  ${ADD_CONTROLFLOW_CONDITION_SRC_FILES}
  RemoveAttributes.cpp

  DEPENDS
  AddDynamicCVPipelinePassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
  BiShengIRAnnotationDialect
  MLIRLinalgTransformOps
  MLIRLinalgTransforms
  MLIRSCFTransformOps
  MLIRTransformDialect
  MLIRTransformDialectTransforms
  MLIRTransforms
)
