include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
# [metax+CommonIR] 剥掉全局 __TLE__：nvidia lowering 按 metax fork 的非-TLE 头编译（回到原可编状态）
if(FLAGTREE_BACKEND STREQUAL "metax")
  remove_definitions(-D__TLE__)
  list(REMOVE_ITEM LLVM_TABLEGEN_FLAGS -D__TLE__)
endif()
add_subdirectory(include)
add_subdirectory(lib)
if(FLAGTREE_TLE AND NOT FLAGTREE_BACKEND STREQUAL "metax")
  add_subdirectory(tle_raw)
endif()
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonNVIDIA ${CMAKE_CURRENT_SOURCE_DIR}/triton_nvidia.cc LINK_LIBS TritonNVIDIAGPUToLLVM NVGPUToLLVM)
  add_dependencies(TritonNVIDIA
    NVGPUTableGen
    NVGPUAttrDefsIncGen
    NVWSTableGen
    NVWSAttrDefsIncGen
    NVWSTransformsIncGen
    TritonNVIDIAGPUConversionPassIncGen
    NVGPUConversionPassIncGen
    TritonNvidiaGPUTableGen
    TritonNvidiaGPUAttrDefsIncGen
    TritonNvidiaGPUOpInterfacesIncGen
    TritonNvidiaGPUTransformsIncGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUCTAAttrIncGen
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen
  )
  if(FLAGTREE_COMMON_IR_ENABLED AND TARGET CommonIRToTTGIR)
    target_link_libraries(TritonNVIDIA PRIVATE CommonIRToTTGIR)
  endif()
  target_link_libraries(TritonNVIDIA PRIVATE Python3::Module pybind11::headers)
  if(FLAGTREE_TLE AND NOT FLAGTREE_BACKEND STREQUAL "metax")
    target_link_libraries(TritonNVIDIA PRIVATE NvidiaTLERawTransforms)
    add_dependencies(TritonNVIDIA
      NvidiaTLERawTransformsIncGen
      TleTableGen
    )
  endif()
endif()
if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
add_subdirectory(hopper)
if(TARGET TritonNVIDIA)
  add_dependencies(TritonNVIDIA NVHopperTransformsIncGen)
endif()
