get_flagtree_backend_lib("TritonGPUIR" _EXTRA_LINK_LIBS)

add_triton_library(TritonGPUIR
  Dialect.cpp
  LinearLayoutConversions.cpp
  Types.cpp

  DEPENDS
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  TritonIR
  TritonTools
  ${_EXTRA_LINK_LIBS}
)
