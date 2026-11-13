add_triton_library(MetaxMmaLayout
  MmaLayout.cpp

  LINK_LIBS PUBLIC
  CommonIRIR
  TritonIR
  TritonGPUIR
  TritonAnalysis
  TritonTools
)

add_triton_library(MetaxCommonIRToTTGIR
  CommonIRToTTGIR.cpp
  CommonIRSetLayout.cpp
  CommonIRLowerTensorTiles.cpp

  DEPENDS
  MetaxCommonIRToTTGIRPassIncGen

  LINK_LIBS PUBLIC
  MetaxMmaLayout
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
  target_include_directories(MetaxCommonIRToTTGIR PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/metax/plugin
    ${PROJECT_BINARY_DIR}/third_party/metax/plugin
  )
endif()

# Layout planning uses TTGIR transforms. Keep it separate from the conversion
# library consumed by TritonToTritonGPU, which those transforms depend on.
add_triton_library(MetaxCommonIRLayouts
  AsyncCopyUtils.cpp
  CommonIRLegalizeAsyncCopy.cpp
  ResolveBsmLayouts.cpp
  CommonIROptimizeSharedLayouts.cpp
  CommonIROptimizeTensorTileLayouts.cpp
  SharedLayoutUtils.cpp
  DotLayoutPlanner.cpp
  TensorLayoutRules.cpp

  DEPENDS
  MetaxCommonIRToTTGIRPassIncGen

  LINK_LIBS PUBLIC
  MetaxMmaLayout
  CommonIRAxisInfo
  CommonIRTransforms
  MLIRAnalysis
  MLIRSCFDialect
  MLIRTransforms
  TritonGPUIR
  TritonGPUTransforms
  TritonAnalysis
  TritonTools
)
