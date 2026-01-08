# flagtree tle
# IR directory for TLE dialect
# Add IR-related source files here
add_triton_library(TleIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TleTableGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)
