include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/spec_cpp)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/spec_cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/spec_cpp/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/spec_cpp/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/musa/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/musa/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tle/dialect/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/tle/dialect/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/tle/frontend/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/tle/frontend/include)
if(FLAGTREE_MTHREADS_TLE)
  # The mthreads TLE frontend emits the backend-independent tle.pipe ops, but
  # the standalone TLE plugin is intentionally disabled for mthreads builds.
  # Build only the common TLE IR contract needed by the backend adapter.
  add_subdirectory(
    ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/include/IR
    ${CMAKE_BINARY_DIR}/third_party/tle/dialect/include/IR)
  add_subdirectory(
    ${CMAKE_SOURCE_DIR}/third_party/tle/dialect/lib/IR
    ${CMAKE_BINARY_DIR}/third_party/tle/dialect/lib/IR)
  add_subdirectory(tle)
  add_dependencies(TritonToTritonGPU MUSATLETableGen)
  target_link_libraries(TritonToTritonGPU PUBLIC MUSATLEIR)
endif()

add_subdirectory(spec_cpp/include/triton/Dialect/Triton/IR)
add_subdirectory(spec_cpp/include/triton/Dialect/TritonGPU/IR)
add_subdirectory(spec_cpp/include/triton/Dialect/TritonNvidiaGPU/IR)
add_subdirectory(spec_cpp/include/triton/Dialect/NVGPU/IR)
add_subdirectory(spec_cpp/include/triton/Dialect/NVWS/IR)
add_subdirectory(spec_cpp/include/triton/Dialect/NVWS/Transforms)

add_subdirectory(spec_cpp/lib/Dialect/NVGPU/IR)
add_subdirectory(spec_cpp/lib/Dialect/NVWS/IR)
add_subdirectory(spec_cpp/lib/Dialect/NVWS/Transforms)

set(MTHREADS_SPEC_CPP_LIB
    ${CMAKE_CURRENT_SOURCE_DIR}/spec_cpp/lib)

target_sources(TritonAnalysis PRIVATE
  ${MTHREADS_SPEC_CPP_LIB}/Analysis/BufferRegion.cpp)
target_sources(TritonGPUToLLVM PRIVATE
  ${MTHREADS_SPEC_CPP_LIB}/Conversion/TritonGPUToLLVM/WarpSpecializeUtility.cpp)
target_sources(TritonGPUTransforms PRIVATE
  ${MTHREADS_SPEC_CPP_LIB}/Dialect/TritonGPU/Transforms/WarpSpecialization/PartitionSchedulingUtility.cpp)
target_sources(TritonNvidiaGPUTransforms PRIVATE
  ${MTHREADS_SPEC_CPP_LIB}/Dialect/TritonNvidiaGPU/Transforms/ProxyFenceInsertion.cpp)
target_sources(TritonLLVMIR PRIVATE
  ${MTHREADS_SPEC_CPP_LIB}/Target/LLVMIR/LLVMDIUtils.cpp)
target_sources(TritonTools PRIVATE
  ${MTHREADS_SPEC_CPP_LIB}/Tools/PluginUtils.cpp)

set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/lib/Dialect/TritonGPU/Transforms/ProcessSharedMemoryHint.cpp
  TARGET_DIRECTORY TritonGPUTransforms
  PROPERTIES HEADER_FILE_ONLY ON)
set_source_files_properties(
  ${PROJECT_SOURCE_DIR}/lib/Dialect/TritonNvidiaGPU/Transforms/ProxFenceInsertion.cpp
  TARGET_DIRECTORY TritonNvidiaGPUTransforms
  PROPERTIES HEADER_FILE_ONLY ON)

add_dependencies(TritonTableGen TritonTypeInterfacesIncGen)
add_dependencies(TritonGPUTableGen TritonGPUOpsEnumsIncGen)
add_dependencies(TritonGPUAttrDefsIncGen
  TritonGPUCGAAttrIncGen
  TritonGPUOpsEnumsIncGen)
add_dependencies(TritonNvidiaGPUTableGen TritonNvidiaGPUTypesIncGen)

if(TARGET PrintLoadStoreMemSpaces)
  set_target_properties(PrintLoadStoreMemSpaces
    PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

add_subdirectory(musa)
if(TRITON_BUILD_PYTHON_MODULE)
  if(FLAGTREE_MTHREADS_TLE)
    set(_MTHREADS_TLE_PLUGIN_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/tle/dialect/triton_mthreads_tle.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/tle/frontend/triton_mthreads_frontend.cc)
    set(_MTHREADS_TLE_PLUGIN_LIBS
        MUSATLEIR MUSATLETransforms MUSATLEFrontendTransforms TleIR)
    set(_MTHREADS_TLE_PLUGIN_DEPS
        MUSATLETableGen
        MUSATLETransforms
        MUSATLEFrontendTransformsIncGen
        MUSATLEFrontendTransforms
        TleTableGen)
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
