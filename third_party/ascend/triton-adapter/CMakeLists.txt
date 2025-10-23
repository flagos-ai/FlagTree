# Security compilation options settings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIE")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIE")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-strong")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fstack-protector-strong")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,now -pie -s")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-z,now -s")
set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
unset(CMAKE_INSTALL_RPATH)

option(TRITON_ADAPTER_BUILD_CPU_BACKEND "Build triton-adapter CPU backend" ON)

set(TRITON_ADAPTER_SOURCE_DIR ".")
set(TRITON_ADAPTER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

include_directories(./include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files

# Triton Adaptor is dependent on AscendNPU IR
set(ASCENDNPU_IR_SRC_DIR ${PROJECT_SOURCE_DIR}/third_party/ascend/third_party/ascendnpu-ir)
set(ASCENDNPU_IR_BINARY_DIR ${PROJECT_BINARY_DIR}/third_party/ascend/third_party/ascendnpu-ir)
set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)

add_subdirectory(${ASCENDNPU_IR_SRC_DIR} ${ASCENDNPU_IR_BINARY_DIR})
include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include) # Tablegen'd files

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if (TRITON_ADAPTER_BUILD_CPU_BACKEND)
    add_triton_plugin(TritonAdapter triton_adapter.cc LINK_LIBS TritonToLinalg)
endif()
