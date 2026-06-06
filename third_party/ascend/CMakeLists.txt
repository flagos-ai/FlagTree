include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party/flir/include)
include_directories(${CMAKE_BINARY_DIR}/third_party/flir/include)

# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Triton Ascend is dependent on AscendNPU IR
# set(ASCENDNPU_IR_SRC_DIR "${PROJECT_SOURCE_DIR}/third_party/ascend/AscendNPU-IR")
# set(ASCENDNPU_IR_BINARY_DIR "${PROJECT_BINARY_DIR}/third_party/ascend/AscendNPU-IR")

# set(BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES ON)
# set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)
#
# # Temporarily save and clear the RULE_LAUNCH_* attributes in the current directory to ensure AscendNPU-IR is not affected.
# get_property(_saved_launch_compile DIRECTORY PROPERTY RULE_LAUNCH_COMPILE)
# get_property(_saved_launch_link DIRECTORY PROPERTY RULE_LAUNCH_LINK)
# set_property(DIRECTORY PROPERTY RULE_LAUNCH_COMPILE "")
# set_property(DIRECTORY PROPERTY RULE_LAUNCH_LINK "")
#
# add_subdirectory(${ASCENDNPU_IR_SRC_DIR} ${ASCENDNPU_IR_BINARY_DIR})
#
# # restore properties
# set_property(DIRECTORY PROPERTY RULE_LAUNCH_COMPILE ${_saved_launch_compile})
# set_property(DIRECTORY PROPERTY RULE_LAUNCH_LINK ${_saved_launch_link})
#
# include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
# include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include) # Tablegen'd files

add_subdirectory(backend/spec/lib)

add_subdirectory(${PROJECT_SOURCE_DIR}/include
                 ${PROJECT_BINARY_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/lib
                 ${PROJECT_BINARY_DIR}/lib)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(bin)

add_triton_library(Registrar Registrar.cc)

if(LLVM_MAJOR_VERSION_21_COMPATIBLE)
  set(_MLIRMeshDialect_LIB MLIRMeshDialect)
else()
  set(_MLIRMeshDialect_LIB "")
endif()

add_triton_plugin(TritonAscend
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ascend_ir.cc
  LINK_LIBS
  TritonToLinalgIncubated
  BiShengIRScopeDialect
  BiShengIRHIVMDialect
  ${_MLIRMeshDialect_LIB}
)

option(TRITON_ENABLE_COVERAGE_LLVM_COV "Enable code llvm-cov coverage tool for Ascend plugin " OFF)
if(TRITON_ENABLE_COVERAGE_LLVM_COV)
  message(STATUS "Enabling llvm-cov coverage tool flags for TritonAscend")
  target_compile_options(TritonAscend PRIVATE
    -fprofile-arcs
    -ftest-coverage
    -O0
    -fprofile-update=atomic
    --coverage
  )
  target_link_options(TritonAscend PRIVATE
    --coverage
    -lgcov
  )
  # branch coverage
  target_compile_definitions(TritonAscend PRIVATE
    COVERAGE_ENABLED=1
  )
endif()


# To enable hitest coverage tool
if(TRITON_ENABLE_COVERAGE_HITEST)
  set_target_properties(TritonAscend PROPERTIES
    RULE_LAUNCH_COMPILE "hitestwrapper"
    RULE_LAUNCH_LINK "hitestwrapper"
  )
endif()

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()
