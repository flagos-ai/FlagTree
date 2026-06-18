# Copyright 2026- Xcoresigma Technology Co., Ltd

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
# TileIR dialect (FLIR) headers — tle_ir.cc creates tile.* ops
include_directories(${CMAKE_SOURCE_DIR}/third_party/flir/include)
include_directories(${CMAKE_BINARY_DIR}/third_party/flir/include)
add_subdirectory(dialect)

# DSA is packaged into the main TritonTLE plugin. Keep a single Python module
# and a single tle dialect registration path.
#if (TRITON_BUILD_PYTHON_MODULE)
#  add_triton_plugin(TritonTLEDSA
#    ${CMAKE_CURRENT_SOURCE_DIR}/tle_ir.cc
#
#    LINK_LIBS
#    TleDSAIR
#    TritonIR
#  )
#
#  find_package(Python3 REQUIRED COMPONENTS Development Interpreter)
#  find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")
#  include_directories(${Python3_INCLUDE_DIRS})
#  include_directories(${pybind11_INCLUDE_DIR})
#  link_directories(${Python3_LIBRARY_DIRS})
#  link_libraries(${Python3_LIBRARIES})
#  add_link_options(${Python3_LINK_OPTIONS})
#endif()
