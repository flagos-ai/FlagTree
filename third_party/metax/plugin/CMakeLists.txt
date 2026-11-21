include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(NOT LLVM_ENABLE_RTTI)
  set_source_files_properties(CopyCacheTranslation.cpp PROPERTIES COMPILE_OPTIONS "-fno-rtti")
endif()
if(TRITON_BUILD_PYTHON_MODULE)
  add_triton_plugin(TritonMETAX ${CMAKE_CURRENT_SOURCE_DIR}/triton_metax.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/CopyCacheTranslation.cpp
    LINK_LIBS TritonMETAXGPUToLLVM MLIRMACADialect MLIRGPUToMACATransforms
      MLIRParser MLIRLLVMToLLVMIRTranslation MLIRBuiltinToLLVMIRTranslation MLIRTargetLLVMIRExport
  )
  add_library(metaxTritonPlugin SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/triton_metax.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/CopyCacheTranslation.cpp
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
    MLIRMACADialect
    MLIRGPUToMACATransforms
  )
  target_sources(metaxTritonPlugin PRIVATE
    $<TARGET_OBJECTS:TritonMETAXGPUToLLVM>
    $<TARGET_OBJECTS:TritonMETAXGPUTransforms>
  )
  target_link_libraries(metaxTritonPlugin PRIVATE
    $<TARGET_FILE:MLIRMACADialect>
    $<TARGET_FILE:MLIRGPUToMACATransforms>
    Python3::Module
    pybind11::headers
    MLIRParser MLIRLLVMToLLVMIRTranslation MLIRBuiltinToLLVMIRTranslation MLIRTargetLLVMIRExport
  )
endif()
