# Copyright 2026- Xcoresigma Technology Co., Ltd

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
add_subdirectory(dialect)

if(FLAGTREE_BACKEND STREQUAL "ascend")
  # CommonIR/flir build settings are consumed by the main TritonTLE plugin.
  include_directories(${CMAKE_SOURCE_DIR}/third_party/flir/include)
  include_directories(${CMAKE_BINARY_DIR}/third_party/flir/include)
  set(TLE_DSA_PLUGIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/tle_ir.cc PARENT_SCOPE)
  set(TLE_DSA_PLUGIN_DEPENDS
    TleDSATableGen
    CommonIRTableGen
    CommonIRAttrDefsIncGen
    CommonIRTypesIncGen
    PARENT_SCOPE)
  set(TLE_DSA_IR_DEPENDS TleDSATableGen PARENT_SCOPE)
  set(TLE_DSA_IR_LINK_LIBS TleDSAIR PARENT_SCOPE)
  set(TLE_DSA_PLUGIN_LINK_LIBS
    TleIR
    CommonIRIR
    PARENT_SCOPE)
  set(TLE_DSA_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/third_party/flir/include
    ${CMAKE_BINARY_DIR}/third_party/flir/include
    PARENT_SCOPE)
endif()

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
