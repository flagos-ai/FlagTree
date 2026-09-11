include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonMETAX 
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_metax.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/Gluon/GluonPythonBindings.cpp
    LINK_LIBS TritonMETAXGPUToLLVM TritonMETAXGPUGluon MLIRMACADialect MLIRGPUToMACATransforms
  )
  add_library(metaxTritonPlugin SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_metax.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/Gluon/GluonPythonBindings.cpp
  )
  set_target_properties(metaxTritonPlugin PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    POSITION_INDEPENDENT_CODE ON
  )
  set_target_properties(metaxTritonPlugin PROPERTIES
    LINK_FLAGS "-Wl,--exclude-libs,ALL"
  )
  add_dependencies(metaxTritonPlugin
    TritonMETAXGPUToLLVM
    TritonMETAXGPUTransforms
    TritonMETAXGPUGluon
    MLIRMACADialect
    MLIRGPUToMACATransforms
  )
  target_sources(metaxTritonPlugin PRIVATE
    $<TARGET_OBJECTS:TritonMETAXGPUToLLVM>
    $<TARGET_OBJECTS:TritonMETAXGPUTransforms>
    $<TARGET_OBJECTS:TritonMETAXGPUGluon>
  )
  target_link_libraries(metaxTritonPlugin PRIVATE
    $<TARGET_FILE:MLIRMACADialect>
    $<TARGET_FILE:MLIRGPUToMACATransforms>
    Python3::Module
    pybind11::headers
  )
endif()
