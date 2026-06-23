# flagtree tle
# IR directory for TLE dialect
# Add IR-related source files here
add_triton_library(TleIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  TleTableGen
  TleDSATableGen

  LINK_LIBS PUBLIC
  TleDSAIR
  TritonIR
  TritonGPUIR
)
