add_triton_library(MetaxCommonIRToTTGIR
  CommonIRToTTGIR.cpp
  CommonIRTensorOps.cpp

  DEPENDS
  MetaxCommonIRToTTGIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRSCFDialect
  MLIRTransforms
  CommonIRIR
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
  CommonIRApplyLayoutPlan.cpp
  CommonIRBsmLayout.cpp
  CommonIRDotLoadLayouts.cpp
  CommonIRSharedLayouts.cpp
  DotLayoutPlanner.cpp

  DEPENDS
  MetaxCommonIRToTTGIRPassIncGen

  LINK_LIBS PUBLIC
  CommonIRToTritonGPU
  MLIRSCFDialect
  TritonGPUIR
  TritonTools
)
