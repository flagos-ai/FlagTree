cmake_minimum_required(VERSION 3.20)

project(flagtree_evas LANGUAGES C CXX)

include(FetchContent)

set(EVAS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(EVAS_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

set(TRITON_SHARED_GIT_REPOSITORY
    "https://github.com/facebookincubator/triton-shared.git")
set(TRITON_SHARED_GIT_TAG "e0c513317d9e7838b00730fd494e3372400e93db")
set(TRITON_SHARED_SOURCE_DIR "${EVAS_BINARY_DIR}/_deps/evas_triton_shared-src")
set(TRITON_SHARED_BINARY_DIR "${EVAS_BINARY_DIR}/third_party/triton_shared")
set(TRITON_SHARED_LLVM_COMPAT_PATCH
    "${EVAS_SOURCE_DIR}/patches/triton-shared-llvm22-compat.patch")

if(NOT EXISTS "${TRITON_SHARED_LLVM_COMPAT_PATCH}")
  message(FATAL_ERROR "missing triton_shared compatibility patch: ${TRITON_SHARED_LLVM_COMPAT_PATCH}")
endif()

if(NOT TARGET TritonSharedAnalysis)
  find_package(Git REQUIRED)
  FetchContent_Declare(
    evas_triton_shared
    GIT_REPOSITORY "${TRITON_SHARED_GIT_REPOSITORY}"
    GIT_TAG "${TRITON_SHARED_GIT_TAG}"
    PATCH_COMMAND
      "${CMAKE_COMMAND}" -E chdir <SOURCE_DIR> /bin/bash -c
      "\"${GIT_EXECUTABLE}\" apply --check \"${TRITON_SHARED_LLVM_COMPAT_PATCH}\" && \"${GIT_EXECUTABLE}\" apply \"${TRITON_SHARED_LLVM_COMPAT_PATCH}\" || \"${GIT_EXECUTABLE}\" apply --reverse --check \"${TRITON_SHARED_LLVM_COMPAT_PATCH}\"")
  FetchContent_GetProperties(evas_triton_shared)
  if(NOT evas_triton_shared_POPULATED)
    FetchContent_Populate(evas_triton_shared)
  endif()

  set(TRITON_SHARED_SOURCE_DIR "${evas_triton_shared_SOURCE_DIR}")
  add_subdirectory("${TRITON_SHARED_SOURCE_DIR}" "${TRITON_SHARED_BINARY_DIR}")
endif()

set(TRITON_SHARED_TOOLS_DIR "${TRITON_SHARED_SOURCE_DIR}/tools/triton-shared-opt")

include_directories(${EVAS_SOURCE_DIR}/include)
include_directories(${EVAS_BINARY_DIR}/include)
include_directories(${EVAS_SOURCE_DIR}/backend/include)
include_directories(${TRITON_SHARED_SOURCE_DIR}/include)
include_directories(${TRITON_SHARED_BINARY_DIR}/include)

add_subdirectory(include/evas/Dialect/Linalg/IR)
add_subdirectory(include/evas/Transform/Linalg)

add_subdirectory(lib/Dialect/Linalg/IR)
add_subdirectory(lib/Transform/Linalg)
add_subdirectory(lib/Conversion/TritonToEvas)
add_subdirectory(bin/evas-triton-opt)

if(TRITON_BUILD_PYTHON_MODULE AND COMMAND add_triton_plugin)
  add_triton_plugin(TritonEVAS ${EVAS_SOURCE_DIR}/backend/triton_evas.cc)
  target_link_libraries(TritonEVAS PRIVATE Python3::Module pybind11::headers)
endif()
