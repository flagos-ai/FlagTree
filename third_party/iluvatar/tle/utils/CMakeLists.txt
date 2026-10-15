# TLE-Raw: mirrors trunk's third_party/tle/utils (TritonTLERawUtils).
add_triton_library(IluvatarTleRawUtils
  lib/Protocol.cpp
  lib/AnalyzeReturnType.cpp
  lib/TleRawMaterialize.cpp

  DEPENDS
  IluvatarTleTableGen

  LINK_LIBS PUBLIC
  IluvatarTleIR
  TritonIR
  MLIRTargetLLVMIRImport
)

target_include_directories(IluvatarTleRawUtils PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../dialect/include
  ${CMAKE_CURRENT_BINARY_DIR}/../dialect/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}/include
  ${PROJECT_SOURCE_DIR}/python/src
)

# These TUs pull in ir.h (and therefore pybind11) to reach TritonOpBuilder,
# while add_triton_library inherits LLVM's -fno-exceptions -fno-rtti.
target_compile_options(IluvatarTleRawUtils PRIVATE -fexceptions -frtti)
