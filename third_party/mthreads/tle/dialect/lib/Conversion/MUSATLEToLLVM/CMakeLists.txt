if(FLAGTREE_MTHREADS_TLE)
  set(_MUSATLE_TILE_TO_LLVM_SOURCES
      TileOpUtils.cpp
      TileOpsToLLVM.cpp)
else()
  set(_MUSATLE_TILE_TO_LLVM_SOURCES "")
endif()

add_triton_library(MUSATLEToLLVM
  LocalPointersOpToLLVM.cpp
  ${_MUSATLE_TILE_TO_LLVM_SOURCES}

  DEPENDS
  MUSATLETableGen

  LINK_LIBS PUBLIC
  MUSATLEIR
  TritonGPUToLLVM
  TritonGPUIR
)
