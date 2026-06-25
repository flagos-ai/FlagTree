# flagtree tle
# IR directory for TLE dialect
# Add IR-related source files here
add_triton_library(TleIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TleTableGen
  ${TLE_DSA_IR_DEPENDS}

  LINK_LIBS PUBLIC
  ${TLE_DSA_IR_LINK_LIBS}
  TritonIR
  TritonGPUIR
)
