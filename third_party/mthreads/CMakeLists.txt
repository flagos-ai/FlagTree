include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/musa/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/musa/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tle/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/tle/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tle/frontend/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/tle/frontend/include)
if(FLAGTREE_MTHREADS_TLE)
  add_subdirectory(tle)
endif()
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(musa)
if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_MTHREADS_TLE)
    set(_MTHREADS_TLE_PLUGIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tle/dialect/triton_mthreads_tle.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tle/frontend/triton_mthreads_frontend.cc)
    set(_MTHREADS_TLE_PLUGIN_LIBS
        MUSATLEIR MUSATLETransforms MUSATLEFrontendTransforms)
    set(_MTHREADS_TLE_PLUGIN_DEPS
        MUSATLETableGen
        MUSATLETransforms
        MUSATLEFrontendTransformsIncGen
        MUSATLEFrontendTransforms)
  else()
    set(_MTHREADS_TLE_PLUGIN_SOURCES "")
    set(_MTHREADS_TLE_PLUGIN_LIBS "")
    set(_MTHREADS_TLE_PLUGIN_DEPS "")
  endif()
  add_triton_plugin(TritonMthreads ${CMAKE_CURRENT_SOURCE_DIR}/triton_mthreads.cc
                              ${_MTHREADS_TLE_PLUGIN_SOURCES}
                    LINK_LIBS TritonMUSAGPUToLLVM MTGPUToLLVM
                              TritonMUSAGPUTransforms
                              ${_MTHREADS_TLE_PLUGIN_LIBS}
                              MLIRMTVMToLLVMIRTranslation)
  add_dependencies(TritonMthreads
                   MUSATableGen
                   MUSAAttrDefsIncGen
                   ${_MTHREADS_TLE_PLUGIN_DEPS}
                   MTGPUTableGen
                   MTGPUTypesIncGen
                   MTGPUConversionPassIncGen
                   TritonMUSAGPUConversionPassIncGen
                   TritonMUSAGPUTransformsIncGen)
  target_link_libraries(TritonMthreads PRIVATE Python3::Module pybind11::headers)
endif()
add_subdirectory(bin)
