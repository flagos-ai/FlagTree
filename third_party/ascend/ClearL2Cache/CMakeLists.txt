# Copyright 2026- Xcoresigma Technology Co., Ltd
cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(ClearL2Cache LANGUAGES CXX ASC)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYBIND11_CMAKE_PREFIX_PATH
    ERROR_QUIET
)
find_package(pybind11 REQUIRED HINTS ${PYBIND11_CMAKE_PREFIX_PATH})

pybind11_add_module(clear_l2 SHARED clear_l2.asc)

get_directory_property(my_defs COMPILE_DEFINITIONS)
remove_definitions(-DFLAGTREE_BACKEND="ascend")
add_definitions(-DFLAGTREE_BACKEND=ascend)

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
message("ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")

target_include_directories(clear_l2 PRIVATE
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/include/experiment/msprof
  ${ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
)

target_link_directories(clear_l2 PUBLIC
  ${ASCEND_HOME_PATH}/lib64
)

target_link_libraries(clear_l2 PUBLIC
  runtime
  ascendcl
)

set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "NPU architecture: dav-2201, dav-3510")
target_compile_options(clear_l2 PRIVATE
    ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
