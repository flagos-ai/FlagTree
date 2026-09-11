add_triton_library(TritonMETAXGPUGluon
  Analysis/GluonDotAnalysis.cpp
  GluonKernelReplay.cpp
  GluonLayoutPlaceholders.cpp
  Analysis/GluonLayoutPropagation.cpp
  Analysis/GluonMemDescAliasAnalysis.cpp
  Analysis/GluonRegionBranchAnalysis.cpp
  GluonSharedAccessPath.cpp
  GluonC500LayoutHelpers.cpp
  GluonC500AsyncCopyLayout.cpp
  GluonLayoutCandidate.cpp
  Targets/GluonC500LayoutRules.cpp
  Targets/GluonC500RegisterSlice.cpp
  InsertGvmArriveBarrierShared.cpp
  GluonInsertRequireLayout.cpp
  GluonLegalizeC500AsyncCopyLayout.cpp
  GluonLegalizeRegisterSlices.cpp
  GluonPropagateLayout.cpp
  GluonRegisterToShared.cpp
  GluonReorderInstructions.cpp
  GluonResolvePlaceholderLayouts.cpp
  GluonLegalizeLocalDotStaging.cpp
  GluonSelectC500LayoutTransfers.cpp
  Targets/GluonC500Layout.cpp
  GluonVerifyLayoutContracts.cpp

  DEPENDS
  TritonMETAXGPUGluonIncGen
  TritonGPUIR

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
  TritonAnalysis
  TritonTransforms
  GluonIR
  GluonTransforms
  MLIRAnalysis
  MLIRTransforms
  TritonGPUTransforms
  TritonMETAXGPUTransforms
)

target_include_directories(TritonMETAXGPUGluon PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(TritonMETAXGPUGluon PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/../../include)
