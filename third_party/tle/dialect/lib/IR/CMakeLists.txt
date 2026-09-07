# flagtree tle
# IR directory for TLE dialect
# Add IR-related source files here
set(TLE_IR_SRC
  Dialect.cpp
  Ops.cpp
  VerifyUtils.cpp
)

if(FLAGCX_ENABLED)
 list(APPEND TLE_IR_SRC
  FlagCxOps.cpp
)
endif()

add_triton_library(TleIR
  ${TLE_IR_SRC}

  DEPENDS
  TleTableGen
  TritonGPUTableGen
  TritonGPUAttrDefsIncGen
  TritonGPUCTAAttrIncGen
  TritonGPUTypeInterfacesIncGen
  TritonGPUOpInterfacesIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
