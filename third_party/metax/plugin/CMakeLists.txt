include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
add_subdirectory(include)
add_subdirectory(lib)
if(FLAGTREE_NATIVE_CACHE_POLICY)
  if(NOT LLVM_ENABLE_RTTI)
    set_source_files_properties(CopyCacheTranslation.cpp PROPERTIES COMPILE_OPTIONS "-fno-rtti")
  endif()
  if(TRITON_BUILD_PYTHON_MODULE)
    add_triton_plugin(TritonMETAX ${CMAKE_CURRENT_SOURCE_DIR}/triton_metax.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/CopyCacheTranslation.cpp
      LINK_LIBS TritonMETAXGPUToLLVM MLIRMACADialect MLIRGPUToMACATransforms
        MLIRParser MLIRLLVMToLLVMIRTranslation MLIRBuiltinToLLVMIRTranslation MLIRTargetLLVMIRExport
    )
  endif()

  if(FLAGTREE_PLUGIN AND TRITON_BUILD_PYTHON_MODULE)
    add_library(metaxTritonPlugin SHARED
      ${CMAKE_CURRENT_SOURCE_DIR}/triton_metax.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/CopyCacheTranslation.cpp
    )
    set_target_properties(metaxTritonPlugin PROPERTIES
      PREFIX ""
      OUTPUT_NAME "metaxTritonPlugin"
      LIBRARY_OUTPUT_DIRECTORY "${TRITON_WHEEL_DIR}/_C"
    )
    target_link_options(metaxTritonPlugin PRIVATE
      -Wl,--exclude-libs,ALL
    )
    target_link_libraries(metaxTritonPlugin PRIVATE
      $<TARGET_FILE:TritonMETAXGPUTransforms>
      $<TARGET_FILE:TritonMETAXGPUToLLVM>
      $<TARGET_FILE:MLIRMACADialect>
      $<TARGET_FILE:MLIRGPUToMACATransforms>
      Python3::Module
      pybind11::headers
      MLIRParser MLIRLLVMToLLVMIRTranslation MLIRBuiltinToLLVMIRTranslation MLIRTargetLLVMIRExport
    )
  endif()
else()
  if(TRITON_BUILD_PYTHON_MODULE)
    add_triton_plugin(TritonMETAX ${CMAKE_CURRENT_SOURCE_DIR}/triton_metax.cc
      LINK_LIBS TritonMETAXGPUToLLVM MLIRMACADialect MLIRGPUToMACATransforms
    )
    add_library(metaxTritonPlugin SHARED
      ${CMAKE_CURRENT_SOURCE_DIR}/triton_metax.cc
    )
    set_target_properties(metaxTritonPlugin PROPERTIES
      PREFIX ""
      OUTPUT_NAME "metaxTritonPlugin"
      LIBRARY_OUTPUT_DIRECTORY "${TRITON_WHEEL_DIR}/_C"
    )
    target_link_options(metaxTritonPlugin PRIVATE
      -Wl,--exclude-libs,ALL
    )
    target_link_libraries(metaxTritonPlugin PRIVATE
      $<TARGET_FILE:TritonMETAXGPUTransforms>
      $<TARGET_FILE:TritonMETAXGPUToLLVM>
      $<TARGET_FILE:MLIRMACADialect>
      $<TARGET_FILE:MLIRGPUToMACATransforms>
      Python3::Module
      pybind11::headers
    )
  endif()
endif()
