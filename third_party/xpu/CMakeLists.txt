function(add_xpu_sdnn_object name)
  set_property(GLOBAL APPEND PROPERTY TRITON_LIBS ${name})
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT IMPORTED GLOBAL)
  message(STATUS "DEBUG add_xpu_sdnn_object ARG_UNPARSED_ARGUMENTS [${ARG_UNPARSED_ARGUMENTS}]")
  set_target_properties(${name} PROPERTIES
    IMPORTED_OBJECTS "${ARG_UNPARSED_ARGUMENTS}"
  )
  if(ARG_DEPENDS)
    add_dependencies(${name} ${ARG_DEPENDS})
  endif()
  if(ARG_LINK_LIBS)
    target_link_libraries(${name} INTERFACE ${ARG_LINK_LIBS})
  endif()
  llvm_update_compile_flags(${name})
endfunction()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include) # For #include "Dialect/TritonXPU/IR/Dialect.h"
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(device)
add_subdirectory(include)
add_subdirectory(lib)

add_library(TritonShareForXPU STATIC IMPORTED)
set_target_properties(TritonShareForXPU PROPERTIES
    IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libTritonSharedForXPU.a
)
target_link_libraries(TritonShareForXPU INTERFACE
    MLIRSCFTransforms
    MLIRLinalgTransforms
    MLIRReconcileUnrealizedCasts
    MLIRSCFTransformOps
    MLIRTransformDialect
    MLIRTransformDialectTransforms
    MLIRLinalgTransformOps
    MLIRTransformDebugExtension
)

if(TRITON_BUILD_PYTHON_MODULE)
    add_triton_plugin(TritonXPU
      ${CMAKE_CURRENT_SOURCE_DIR}/triton_xpu.cc

      LINK_LIBS
      TritonShareForXPU
    ) #LINK_LIBS TritonToTritonXPU TritonXPUToLLVM
    add_xpu_sdnn_object(TritonXPUSDNN
      ${CMAKE_CURRENT_SOURCE_DIR}/triton_xpu_sdnn.cc.o
    )
endif()
