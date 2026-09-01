include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/python/src)

add_subdirectory(include/IR)
add_subdirectory(include/Transforms)

set(ILUVATAR_TLE_IR_SRC
  lib/IR/Dialect.cpp
  lib/IR/Ops.cpp
  lib/IR/VerfiyUtils.cpp
)

set(ILUVATAR_TLE_TO_LLVM_SRC
  lib/Conversion/TleToLLVM.cpp
  lib/Conversion/TleToLLVM/TleTileToLLVMUtils.cpp
  lib/Conversion/TleToLLVM/DistributedBarrierOpToLLVM.cpp
  lib/Conversion/TleToLLVM/ExtractOpToLLVM.cpp
  lib/Conversion/TleToLLVM/ExtractTileToLLVM.cpp
  lib/Conversion/TleToLLVM/InsertTileToLLVM.cpp
  lib/Conversion/TleToLLVM/LocalPointersOpToLLVM.cpp
  lib/Conversion/TleToLLVM/PackOpToLLVM.cpp
)

if(FLAGCX_ENABLED)
  list(APPEND ILUVATAR_TLE_IR_SRC
    lib/IR/FlagCxOps.cpp
  )
  list(APPEND ILUVATAR_TLE_TO_LLVM_SRC
    lib/Conversion/TleToLLVM/GetDeviceIdToFlagCX.cpp
  )
endif()

add_triton_library(IluvatarTleIR
  ${ILUVATAR_TLE_IR_SRC}

  DEPENDS
  IluvatarTleTableGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)

add_triton_library(IluvatarTleTools
  lib/Tools/FlagcxUtils.cpp

  DEPENDS
  IluvatarTleTableGen
)

if(FLAGCX_ENABLED)
  add_triton_library(IluvatarFlagCxToLLVM
    lib/Conversion/TleToLLVM/FlagCxOpToLLVM/DeviceIntraBarrierOpToLLVM.cpp
    lib/Conversion/TleToLLVM/FlagCxOpToLLVM/FlagCxOpToLLVM.cpp
    lib/Conversion/TleToLLVM/FlagCxOpToLLVM/GetLocalRankOpToLLVM.cpp

    DEPENDS
    IluvatarTleTableGen
    TritonGPUTypeInterfacesIncGen
    IluvatarTleTools

    LINK_LIBS PUBLIC
    IluvatarTleIR
    IluvatarTleTools
    TritonGPUToLLVM
  )
endif()

add_triton_library(IluvatarTleToLLVM
  ${ILUVATAR_TLE_TO_LLVM_SRC}

  DEPENDS
  IluvatarTleTableGen
  TritonGPUTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  IluvatarTleIR
  TritonGPUToLLVM
)

if(FLAGCX_ENABLED)
  target_link_libraries(IluvatarTleToLLVM PUBLIC IluvatarFlagCxToLLVM)
endif()

add_triton_library(IluvatarTleTransforms
  lib/Transforms/TleAddDistributedParams.cpp
  lib/Transforms/ConvertArgToMemDesc.cpp
  lib/Transforms/DSLRegionInline.cpp
  lib/Transforms/TleEarlyAssignMemorySpace.cpp
  lib/Transforms/TleInsertLocalPointerBarriers.cpp
  lib/Transforms/TleLowerAsyncLoad.cpp
  lib/Transforms/TleLowerExclusiveCumsum.cpp
  lib/Transforms/TleMarkSmeDotOperands.cpp
  lib/Transforms/TleOptimizeExclusiveCumsumLayouts.cpp
  lib/Transforms/TleOptimizeLocalPointerAsyncStores.cpp
  lib/Transforms/TleOptimizeLocalPointerLoads.cpp
  lib/Transforms/TleOptimizeLocalPointerStores.cpp
  lib/Transforms/TlePromoteLocalStoreStaging.cpp
  lib/Transforms/TleLowerPipeToBarriers.cpp
  lib/Transforms/RemoveRedundantCopy.cpp
  lib/Transforms/TleUtility.cpp

  DEPENDS
  IluvatarTleTableGen
  IluvatarTleTransformsIncGen

  LINK_LIBS PUBLIC
  IluvatarTleIR
  TritonIR
  TritonGPUIR
  TritonGPUTransforms
  TritonAnalysis
)
