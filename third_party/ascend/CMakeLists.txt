add_subdirectory(backend/spec/lib)
add_subdirectory(${PROJECT_SOURCE_DIR}/include
                 ${PROJECT_BINARY_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib
                 ${PROJECT_BINARY_DIR}/lib)

add_triton_library(Registrar Registrar.cc)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Triton Ascend is dependent on AscendNPU IR
#set(ASCENDNPU_IR_SRC_DIR "${PROJECT_SOURCE_DIR}/third_party/ascendnpu-ir")
#set(ASCENDNPU_IR_BINARY_DIR "${PROJECT_BINARY_DIR}/third_party/ascendnpu-ir")

set(BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES ON)
set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)

#add_subdirectory(${ASCENDNPU_IR_SRC_DIR} ${ASCENDNPU_IR_BINARY_DIR})
#include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
#include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include) # Tablegen'd files

#add_subdirectory(include)
#add_subdirectory(lib)

add_triton_plugin(TritonAscend
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ascend_ir.cc

  LINK_LIBS
  TritonToLinalgIncubated
  BiShengIRScopeDialect
  BiShengIRHIVMDialect
)

# target_link_libraries(TritonAscend PRIVATE Python3::Module pybind11::headers)
target_include_directories(TritonAscend PRIVATE
  ${CMAKE_SOURCE_DIR}/third_party/flir/include
  ${CMAKE_BINARY_DIR}/third_party/flir/include)

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
