include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/musa/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/musa/include)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(musa)
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonMthreads ${CMAKE_CURRENT_SOURCE_DIR}/triton_mthreads.cc
                    LINK_LIBS TritonMUSAGPUToLLVM MTGPUToLLVM TritonMUSAGPUTransforms)
  add_dependencies(TritonMthreads
                   MUSATableGen
                   MUSAAttrDefsIncGen
                   MTGPUTableGen
                   MTGPUTypesIncGen
                   MTGPUConversionPassIncGen
                   TritonMUSAGPUConversionPassIncGen
                   TritonMUSAGPUTransformsIncGen)
  target_link_libraries(TritonMthreads PRIVATE Python3::Module pybind11::headers)
endif()
add_subdirectory(bin)
