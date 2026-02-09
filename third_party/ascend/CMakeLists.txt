include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/flir/include)
include_directories(${CMAKE_BINARY_DIR}/third_party/flir/include)

# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Triton Ascend is dependent on AscendNPU IR
#set(ASCENDNPU_IR_SRC_DIR "${PROJECT_SOURCE_DIR}/third_party/ascend/AscendNPU-IR")
#set(ASCENDNPU_IR_BINARY_DIR "${PROJECT_BINARY_DIR}/third_party/ascend/AscendNPU-IR")

#set(BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES ON)
#set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)

#add_subdirectory(${ASCENDNPU_IR_SRC_DIR} ${ASCENDNPU_IR_BINARY_DIR})
#include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
#include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include) # Tablegen'd files

add_subdirectory(backend/spec/lib)

add_subdirectory(${PROJECT_SOURCE_DIR}/include
                 ${PROJECT_BINARY_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib
                 ${PROJECT_BINARY_DIR}/lib)
add_subdirectory(include)
add_subdirectory(lib)

add_triton_library(Registrar Registrar.cc)

add_triton_plugin(TritonAscend
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ascend_ir.cc

  LINK_LIBS
  TritonToLinalgIncubated
  TritonToLLVM
  TritonToHIVM
  TritonToHFusion
  TritonAscendIR
  BiShengIRScopeDialect
  BiShengIRHIVMDialect
  MLIRMeshDialect
)

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
