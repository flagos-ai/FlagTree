add_triton_library(TritonTLERawUtils
  lib/Protocol.cpp
  lib/AnalyzeReturnType.cpp
  lib/TleRawMaterialize.cpp

  LINK_LIBS PUBLIC
  TleIR
  TritonIR
  MLIRTargetLLVMIRImport
)

target_include_directories(TritonTLERawUtils PUBLIC
  ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/include
  ${CMAKE_SOURCE_DIR}/third_party/tle/utils/include
  ${CMAKE_SOURCE_DIR}/python/src
)

target_compile_options(TritonTLERawUtils PRIVATE -U__TLE__)
