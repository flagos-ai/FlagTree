if(NOT FLAGTREE_BACKEND STREQUAL "iluvatar")
  set(_EXTRA_LINK_LIBS TritonNvidiaGPUIR)
else()
  set(_EXTRA_LINK_LIBS FlagTree_${FLAGTREE_BACKEND}_TritonAnalysis)
endif()

add_triton_library(TritonAnalysis
  AxisInfo.cpp
  Allocation.cpp
  Membar.cpp
  Alias.cpp
  Utility.cpp

  DEPENDS
  TritonTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  ${_EXTRA_LINK_LIBS}
)
