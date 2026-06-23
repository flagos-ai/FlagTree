# flagtree tle
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dsa/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/dsa/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/include)

add_subdirectory(dialect/include)
add_subdirectory(dsa)
add_subdirectory(dialect/lib)
add_subdirectory(dsa/dialect/lib/Conversion)

if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonTLE
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_tle_raw.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/dsa/tle_ir.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/lib/Protocol.cpp
    DEPENDS
    TleDSATableGen
    LINK_LIBS
    MLIRTargetLLVMIRImport
    TritonTLETransforms
    TritonIR)
  target_link_libraries(TritonTLE PRIVATE Python3::Module pybind11::headers MLIRTargetLLVMIRImport)
endif()

add_subdirectory(test)
