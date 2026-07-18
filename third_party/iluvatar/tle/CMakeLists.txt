include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include/IR)
add_subdirectory(include/Transforms)

add_triton_library(IluvatarTleIR
  lib/IR/Dialect.cpp

  DEPENDS
  IluvatarTleTableGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)

add_triton_library(IluvatarTleToLLVM
  lib/Conversion/TleToLLVM.cpp
  lib/Conversion/TleToLLVM/TleTileToLLVMUtils.cpp
  lib/Conversion/TleToLLVM/ExtractTileToLLVM.cpp
  lib/Conversion/TleToLLVM/InsertTileToLLVM.cpp
  lib/Conversion/TleToLLVM/LocalPointersOpToLLVM.cpp

  DEPENDS
  IluvatarTleTableGen
  TritonGPUTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  IluvatarTleIR
  TritonGPUToLLVM
)

add_triton_library(IluvatarTleTransforms
  lib/Transforms/IluvatarTleInsertLocalPointerBarriers.cpp
  lib/Transforms/IluvatarTleOptimizeLocalPointerLoads.cpp
  lib/Transforms/IluvatarTleOptimizeLocalPointerStores.cpp

  DEPENDS
  IluvatarTleTableGen
  IluvatarTleTransformsIncGen

  LINK_LIBS PUBLIC
  IluvatarTleIR
  TritonIR
  TritonGPUIR
)
