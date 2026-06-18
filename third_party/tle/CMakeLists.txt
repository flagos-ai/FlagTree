# flagtree tle
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/include)

if(FLAGTREE_BACKEND STREQUAL "ascend")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dsa/dialect/include)
  include_directories(${CMAKE_CURRENT_BINARY_DIR}/dsa/dialect/include)
  set(TLE_DSA_PLUGIN_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/dsa/tle_ir.cc)
  set(TLE_DSA_PLUGIN_DEPENDS
    TleDSATableGen
    TileIRTableGen
    TileIRAttrDefsIncGen
    TileIRTypesIncGen)
  set(TLE_DSA_IR_DEPENDS TleDSATableGen)
  set(TLE_DSA_IR_LINK_LIBS TleDSAIR)
endif()

add_subdirectory(dialect/include)
if(FLAGTREE_BACKEND STREQUAL "ascend")
  add_subdirectory(dsa)
endif()
add_subdirectory(dialect/lib)
if(FLAGTREE_BACKEND STREQUAL "ascend")
  add_subdirectory(dsa/dialect/lib/Conversion)
endif()

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonTLE
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle_raw.cc
    ${TLE_DSA_PLUGIN_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/lib/Protocol.cpp
    DEPENDS
    ${TLE_DSA_PLUGIN_DEPENDS}
    LINK_LIBS
    MLIRTargetLLVMIRImport
    TleIR
    TileIRIR
    BiShengIRHIVMDialect
    # MLIRMeshDialect
    TritonTLETransforms
    TritonIR)
  if(FLAGTREE_BACKEND STREQUAL "ascend")
    target_include_directories(TritonTLE BEFORE PRIVATE
      ${FLAGTREE_BACKEND_DIR}/python/src)
  endif()
  target_link_libraries(TritonTLE PRIVATE Python3::Module pybind11::headers MLIRTargetLLVMIRImport)
endif()

add_subdirectory(test)
