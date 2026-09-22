add_triton_library(MetaxCommonIRToTTGIR
  CommonIRToTTGIR.cpp
  CommonIRInjectDotPlan.cpp
  DotLayoutPlanner.cpp

  DEPENDS
  MetaxCommonIRToTTGIRPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  CommonIRIR
  TritonIR
  TritonGPUIR
  TritonAnalysis
)

if(BUILD_MCTLE)
  target_include_directories(MetaxCommonIRToTTGIR PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/metax/plugin
    ${PROJECT_BINARY_DIR}/third_party/metax/plugin
  )
endif()
