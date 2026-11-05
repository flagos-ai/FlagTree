# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
# include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
# add_subdirectory(include)
add_subdirectory(lib)
if(TRITON_BUILD_PYTHON_MODULE)
  # find_package(LLD REQUIRED CONFIG PATHS "${LLD_DIR}" NO_DEFAULT_PATH)
  # include_directories(${LLD_INCLUDE_DIRS})
  message(STATUS "Found LLD distro-package @ ${LLD_DIR} and LLD include dirs @ ${LLD_INCLUDE_DIRS}")
  add_triton_plugin(TritonGPNPU ${CMAKE_CURRENT_SOURCE_DIR}/triton_gpnpu.cc)
  target_link_libraries(TritonGPNPU PRIVATE
    Python3::Module
    pybind11::headers
    TritonGPNPUToLLVM
    TritonGPNPUTransforms
    TritonGPUToLLVM
    TritonGPUIR
    TritonNvidiaGPUIR
    NVGPUIR
    TritonIR
    MLIRIR
    MLIRPass
    MLIRLLVMDialect
    MLIRNVVMDialect
    MLIRNVVMToLLVMIRTranslation
    LLVMBitWriter
    LLVMIRReader
    LLVMTransformUtils
  )
  # For MLIR parsing support
  target_link_libraries(TritonGPNPU PRIVATE MLIRParser)

  # External backend packages are linked into python/triton/backends, which
  # the base manifest prunes. Stage non-Python runtime files alongside the
  # extension output so the wheel includes the driver source.
  if(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(GPNPU_RUNTIME_PACKAGE_DIR
        "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/../backends/intellifusion")
    foreach(resource IN ITEMS driver.c)
      configure_file("${CMAKE_CURRENT_SOURCE_DIR}/backend/${resource}"
                     "${GPNPU_RUNTIME_PACKAGE_DIR}/${resource}" COPYONLY)
    endforeach()
  endif()
endif()
# add_subdirectory(test)
