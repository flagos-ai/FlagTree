include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

# Require modern compilers: GCC/Clang version must be >= 13 (i.e., > 13)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 13)
    message(FATAL_ERROR "C++ compiler version must be >= 13 (found ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION})")
  endif()
endif()
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  if(CMAKE_C_COMPILER_VERSION VERSION_LESS 13)
    message(FATAL_ERROR "C compiler version must be >= 13 (found ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION})")
  endif()
endif()

set(HAVE_CUDA_TILE_INSTALL OFF)
if(DEFINED ENV{CUDA_TILE_INSTALL_DIR})
  set(CUDA_TILE_INSTALL_DIR $ENV{CUDA_TILE_INSTALL_DIR})
  set(CUDA_TILE_INCLUDE_DIRS ${CUDA_TILE_INSTALL_DIR}/include/include)
  set(CUDA_TILE_LIBRARY_PATH ${CUDA_TILE_INSTALL_DIR}/lib)
  set(CUDA_TILE_MLIR_INCLUDE_DIRS $ENV{CUDA_TILE_MLIR_INCLUDE_DIRS})
  set(HAVE_CUDA_TILE_INSTALL ON)
else()
  # ===== flagtree-local rewrite of the upstream else() branch =====
  # Upstream does a build-time `git clone https://github.com/NVIDIA/cuda-tile`
  # into ${CMAKE_CURRENT_BINARY_DIR}/tileir_src and applies an inline patch.
  #
  # FlagTree instead:
  #   1) setup.py fetches the pinned cuda-tile checkout under third_party;
  #   2) CMake copies it to the build tree so patches and build artifacts do
  #      not pollute the pinned checkout;
  #   3) CMake applies the LLVM compatibility patch and fails on any mismatch.
  # ================================================================
  # setup.py populates the pinned cuda-tile checkout before CMake runs.
  set(CUDA_TILE_COMMIT 7e8e2e68fa219716103824c01f7303367cf7df8d)
  set(CUDA_TILE_CHECKOUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cuda-tile)
  if(NOT EXISTS ${CUDA_TILE_CHECKOUT_DIR}/CMakeLists.txt)
    message(FATAL_ERROR
      "cuda-tile checkout not populated at ${CUDA_TILE_CHECKOUT_DIR}. Re-run setup.py")
  endif()

  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CUDA_TILE_CHECKOUT_DIR}
    OUTPUT_VARIABLE CUDA_TILE_CHECKOUT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE CUDA_TILE_REV_PARSE_RESULT
  )
  if(NOT CUDA_TILE_REV_PARSE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to read cuda-tile checkout commit at ${CUDA_TILE_CHECKOUT_DIR}")
  endif()
  if(NOT CUDA_TILE_CHECKOUT_COMMIT STREQUAL CUDA_TILE_COMMIT)
    message(FATAL_ERROR
      "cuda-tile checkout is at ${CUDA_TILE_CHECKOUT_COMMIT}, expected ${CUDA_TILE_COMMIT}. "
      "Re-run setup.py to fetch the pinned cuda-tile checkout")
  endif()

  # Copy sources to the build tree so patches and build artifacts do not pollute the checkout.
  set(CUDA_TILE_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/cuda-tile-work)
  set(CUDA_TILE_COPY_MARKER ${CUDA_TILE_SOURCE_DIR}/.flagtree_cuda_tile_commit)
  set(CUDA_TILE_NEEDS_COPY ON)
  if(EXISTS ${CUDA_TILE_SOURCE_DIR}/CMakeLists.txt AND EXISTS ${CUDA_TILE_COPY_MARKER})
    file(READ ${CUDA_TILE_COPY_MARKER} CUDA_TILE_COPIED_COMMIT)
    string(STRIP "${CUDA_TILE_COPIED_COMMIT}" CUDA_TILE_COPIED_COMMIT)
    if(CUDA_TILE_COPIED_COMMIT STREQUAL CUDA_TILE_COMMIT)
      set(CUDA_TILE_NEEDS_COPY OFF)
    endif()
  endif()
  if(CUDA_TILE_NEEDS_COPY)
    message(STATUS "Copying cuda-tile sources from checkout to ${CUDA_TILE_SOURCE_DIR}")
    file(REMOVE_RECURSE ${CUDA_TILE_SOURCE_DIR})
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${CUDA_TILE_CHECKOUT_DIR} ${CUDA_TILE_SOURCE_DIR}
      RESULT_VARIABLE CUDA_TILE_COPY_RESULT
    )
    if(NOT CUDA_TILE_COPY_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to copy cuda-tile sources to ${CUDA_TILE_SOURCE_DIR}")
    endif()
    file(WRITE ${CUDA_TILE_COPY_MARKER} "${CUDA_TILE_COMMIT}\n")
  endif()

  # Patch cuda-tile 13.4 for FlagTree's pinned public LLVM APIs.
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env LLVM_SYSPATH=${LLVM_SYSPATH} bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/patch_bytecode_utils.sh ${CUDA_TILE_SOURCE_DIR}
    WORKING_DIRECTORY ${CUDA_TILE_SOURCE_DIR}
    RESULT_VARIABLE CUDA_TILE_PATCH_RESULT
  )
  if(NOT CUDA_TILE_PATCH_RESULT EQUAL 0)
    message(FATAL_ERROR "cuda_tile patch failed with code: ${CUDA_TILE_PATCH_RESULT}")
  endif()

  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env LLVM_SYSPATH=${LLVM_SYSPATH} LLVM_EXTERNAL_LIT=${LLVM_SYSPATH}/bin/llvm-lit bash ${CMAKE_CURRENT_SOURCE_DIR}/scripts/build_cuda_tile.sh ${CUDA_TILE_SOURCE_DIR}
    WORKING_DIRECTORY ${CUDA_TILE_SOURCE_DIR}
    RESULT_VARIABLE CUDA_TILE_BUILD_RESULT
  )
  if(NOT CUDA_TILE_BUILD_RESULT EQUAL 0)
    message(FATAL_ERROR "cuda_tile build failed with code: ${CUDA_TILE_BUILD_RESULT}")
  endif()

  # After successful build, set variables for downstream usage in this configure
  set(CUDA_TILE_INSTALL_DIR ${CUDA_TILE_SOURCE_DIR}/build/install)
  set(CUDA_TILE_LIBRARY_PATH ${CUDA_TILE_INSTALL_DIR}/lib)
  set(CUDA_TILE_INCLUDE_DIRS ${CUDA_TILE_INSTALL_DIR}/include/include)
  set(CUDA_TILE_MLIR_INCLUDE_DIRS ${CUDA_TILE_SOURCE_DIR}/build/include)
  set(HAVE_CUDA_TILE_INSTALL ON)
  # Export to CACHE and ENV for downstream visibility
  set(CUDA_TILE_INSTALL_DIR ${CUDA_TILE_INSTALL_DIR} CACHE PATH "CUDA Tile install dir" FORCE)
  set(CUDA_TILE_INCLUDE_DIRS ${CUDA_TILE_INCLUDE_DIRS} CACHE PATH "CUDA Tile include dirs" FORCE)
  set(CUDA_TILE_LIBRARY_PATH ${CUDA_TILE_LIBRARY_PATH} CACHE PATH "CUDA Tile library path" FORCE)
  set(HAVE_CUDA_TILE_INSTALL ${HAVE_CUDA_TILE_INSTALL} CACHE BOOL "Has cuda tile install" FORCE)
  set(ENV{CUDA_TILE_INSTALL_DIR} ${CUDA_TILE_INSTALL_DIR})
  set(ENV{CUDA_TILE_INCLUDE_DIRS} ${CUDA_TILE_INCLUDE_DIRS})
  set(ENV{CUDA_TILE_LIBRARY_PATH} ${CUDA_TILE_LIBRARY_PATH})
endif()



# Set up tileir specific variables for lit tests
set(TRITON_BINARY_DIR ${CMAKE_BINARY_DIR})
set(TRITON_CUDA_TILE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(TRITON_CUDA_TILE_TOOL_DIR ${CMAKE_BINARY_DIR}/third_party/tileir/tools/triton-cuda-tile-opt)
set(TRITON_CUDA_TILE_BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
set(TRITON_CUDA_TILE_LIBRARY_DIR ${CMAKE_BINARY_DIR}/third_party/tileir/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

if(HAVE_CUDA_TILE_INSTALL)
  include_directories(${CUDA_TILE_INCLUDE_DIRS})
  include_directories(${CUDA_TILE_MLIR_INCLUDE_DIRS})
endif()

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools/triton-cuda-tile-opt)
add_subdirectory(test)

if(TRITON_BUILD_PYTHON_MODULE AND HAVE_CUDA_TILE_INSTALL)
  add_triton_plugin(TritonTileIR ${CMAKE_CURRENT_SOURCE_DIR}/triton_tileir.cc LINK_LIBS TritonToTileIR TritonTileIRTransforms)
  add_dependencies(TritonTileIR
    TritonToTileIRConversionPassIncGen
    TritonTileIRTransformsIncGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUCTAAttrIncGen
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen
  )
  target_link_libraries(TritonTileIR PRIVATE Python3::Module pybind11::headers)
  target_link_libraries(TritonTileIR PRIVATE ${CUDA_TILE_LIBRARY_PATH}/libCudaTileTransforms.a)
  target_link_libraries(TritonTileIR PRIVATE ${CUDA_TILE_LIBRARY_PATH}/libCudaTileBytecodeWriter.a)
  target_link_libraries(TritonTileIR PRIVATE ${CUDA_TILE_LIBRARY_PATH}/libCudaTileBytecodeCommon.a)
elseif(TRITON_BUILD_PYTHON_MODULE)
  message(STATUS "Skip building TritonTileIR Python plugin: CUDA_TILE_INSTALL_DIR not detected")
endif()
