add_triton_library(MetaxCommonIRToTTGIR
  CommonIRToTTGIR.cpp
  CommonIRSetLayout.cpp
  CommonIRLowerTensorTiles.cpp

  DEPENDS
  MetaxCommonIRToTTGIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTransforms
  CommonIRIR
  CommonIRTransforms
  CommonIRToTritonGPU
  TritonIR
  TritonGPUIR
  TritonAnalysis
  TritonTools
)

if(BUILD_MCTLE)
  add_dependencies(MetaxCommonIRToTTGIR McTleTableGen)
  target_include_directories(MetaxCommonIRToTTGIR PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/metax/plugin
    ${PROJECT_BINARY_DIR}/third_party/metax/plugin
  )
endif()

# Layout planning uses TTGIR transforms. Keep it separate from the conversion
# library consumed by TritonToTritonGPU, which those transforms depend on.
add_triton_library(MetaxCommonIRLayouts
  CommonIRCoalesceAsyncCopy.cpp
  CommonIROptimizeSharedLayouts.cpp
  CommonIROptimizeTensorTileLayouts.cpp
  DotLayoutPlanner.cpp
  TensorTileAxisInfo.cpp

  DEPENDS
  MetaxCommonIRToTTGIRPassIncGen

  LINK_LIBS PUBLIC
  CommonIRTransforms
  MLIRAnalysis
  MLIRSCFDialect
  MLIRTransforms
  TritonGPUIR
  TritonGPUTransforms
  TritonAnalysis
  TritonTools
)
