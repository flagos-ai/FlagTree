# For Triton

# ######################################################
# Parallel jobs for nested cmake --build; use env MAX_JOBS when set, else default -j4
if(DEFINED ENV{MAX_JOBS} AND NOT "$ENV{MAX_JOBS}" STREQUAL "")
  set(JOB_SETTING "-j$ENV{MAX_JOBS}")
else()
  set(JOB_SETTING "-j4")
endif()

# ######################################################
# Get LLVM for triton
include(triton_gcu_llvm)
include(triton_gcu_llvm_config)

# Disable warnings that show up in external code (gtest;pybind11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-but-set-parameter -Wno-attributes")
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files

# 使用本地的 triton 文件，不需要下载（使用根目录的 triton）
set(third_party_triton_${arch}_fetch_src "${CMAKE_SOURCE_DIR}")
set(third_party_triton_${arch}_fetch_bin "${CMAKE_CURRENT_BINARY_DIR}/third_party_triton_${arch}_bin")
file(GLOB_RECURSE third_party_triton_${arch}_src "${CMAKE_SOURCE_DIR}/include/*" "${CMAKE_SOURCE_DIR}/lib/*" "${CMAKE_SOURCE_DIR}/third_party/f2reduce/*" "${CMAKE_SOURCE_DIR}/third_party/proton/*")

include(${CMAKE_CURRENT_LIST_DIR}/triton_${arch}.cmake)

file(MAKE_DIRECTORY ${third_party_triton_${arch}_fetch_bin})

list(APPEND triton_cmake_args -DMLIR_DIR=${MLIR_DIR})
list(APPEND triton_cmake_args -DLLVM_LIBRARY_DIR=${LLVM_LIBRARY_DIR})
list(APPEND triton_cmake_args -DTRITON_BUILD_UT=OFF)
list(APPEND triton_cmake_args -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
list(APPEND triton_cmake_args -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
list(APPEND triton_cmake_args -DCMAKE_BUILD_TYPE=Release)

# 删除${third_party_triton_${arch}_fetch_src}/include/triton/Dialect/Triton/IR/TritonOps.td中HasParent<"ModuleOp">这一行
execute_process(
  COMMAND sed -i "\\#HasParent<\"ModuleOp\">#d"
            ${third_party_triton_${arch}_fetch_src}/include/triton/Dialect/Triton/IR/TritonOps.td
            #ERROR_QUIET
)

# 行首添加return failure(); 对func body增加注释
execute_process(
  COMMAND sed -i "s/\\/\\/ Prevent LLVM's inliner to inline this function/return failure();\\/*/"
            ${third_party_triton_${arch}_fetch_src}/lib/Conversion/TritonGPUToLLVM/FuncOpToLLVM.cpp
            #ERROR_QUIET
)
execute_process(
  COMMAND sed -i "s/return success();/*\\//"
            ${third_party_triton_${arch}_fetch_src}/lib/Conversion/TritonGPUToLLVM/FuncOpToLLVM.cpp
            #ERROR_QUIET
)

# 立即修复 Passes.h 中的 NVWS 依赖问题（移除不存在的 NVIDIA 特定头文件）
execute_process(
    COMMAND sed -i "\\#nvidia/include/Dialect/NVWS/IR/Dialect.h#d"
            ${third_party_triton_${arch}_fetch_src}/include/triton/Dialect/TritonGPU/Transforms/Passes.h
            #ERROR_QUIET
)

# 修复 Passes.td 中的 NVWS dialect 引用（移除 tablegen 定义中的 NVWS）
execute_process(
    COMMAND sed -i "/nvws::NVWSDialect/d"
            ${third_party_triton_${arch}_fetch_src}/include/triton/Dialect/TritonGPU/Transforms/Passes.td
            #ERROR_QUIET
)

add_custom_command(
    OUTPUT  ${triton_${arch}_objs}
    COMMAND sed -i "s/-Wno-covered-switch-default//g" ${third_party_triton_${arch}_fetch_src}/CMakeLists.txt
    COMMAND cmake -S ${third_party_triton_${arch}_fetch_src} -B ${third_party_triton_${arch}_fetch_bin} ${triton_cmake_args} -DTRITON_CODEGEN_BACKENDS='nvidia\;amd' -DCMAKE_CXX_FLAGS='-Wno-reorder -Wno-error=comment' -G Ninja
    COMMAND cmake --build ${third_party_triton_${arch}_fetch_bin} --target all ${JOB_SETTING}
    DEPENDS ${third_party_triton_${arch}_src}
)
set(mlir_register_libs MLIRRegisterAllDialects MLIRRegisterAllExtensions MLIRRegisterAllPasses)

add_custom_target(third_party_triton_${arch}_fetch_build ALL DEPENDS ${triton_${arch}_objs})

add_library(triton_${arch} INTERFACE)
add_dependencies(triton_${arch} third_party_triton_${arch}_fetch_build)

message(STATUS "third_party_triton_${arch}_fetch_bin is {third_party_triton_${arch}_fetch_bin}")

include_directories(${third_party_triton_${arch}_fetch_src}/include)
include_directories(${third_party_triton_${arch}_fetch_bin}/include) # Tablegen'd files

set(MLIR_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

#add_subdirectory(${third_party_triton_${arch}_fetch_src}/include ${third_party_triton_${arch}_fetch_bin}/include)
#add_subdirectory(${third_party_triton_${arch}_fetch_src}/third_party/f2reduce ${third_party_triton_${arch}_fetch_bin}/third_party/f2reduce)

include_directories(${third_party_triton_${arch}_fetch_src})
include_directories(${third_party_triton_${arch}_fetch_bin}/lib/Dialect/Triton/Transforms) # TritonCombine.inc
#add_subdirectory(${third_party_triton_${arch}_fetch_src}/lib ${third_party_triton_${arch}_fetch_bin}/lib)
# include_directories(${CMAKE_CURRENT_BINARY_DIR}/kernels)

add_subdirectory(include)
add_subdirectory(lib)

get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(triton-${arch}-opt triton-${arch}-opt.cpp PARTIAL_SOURCES_INTENDED)
set_target_properties(triton-${arch}-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

llvm_update_compile_flags(triton-${arch}-opt)

# Allow multiple definitions due to nested build generating duplicate symbols
target_link_options(triton-${arch}-opt PRIVATE -Wl,--allow-multiple-definition)

target_link_libraries(triton-${arch}-opt PRIVATE
  GCUIR${arch}
  MemrefExtIR${arch}
  MathExtIR${arch}
  TritonGCUIR_${arch}
  MLIRTritonToGCU_${arch}
  MLIRTritonGCUTransforms_${arch}
  ${mlir_dialect_libs}
  ${mlir_conversion_libs}
  ${mlir_translation_libs}
  ${mlir_extension_libs}
  # TLE libraries (needed by nested build objects)
  TleIR
  TleToLLVM
  TritonTLETransforms
  # Triton GPU transforms (for ProcessSharedMemoryHint pass)
  TritonGPUTransforms
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms
  # MLIR registration (needed by RegisterGCUDialects.h)
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  ${triton_${arch}_objs}
)

add_dependencies(triton-${arch}-opt triton_${arch})

mlir_check_all_link_libraries(triton-${arch}-opt)

target_compile_options(obj.TritonGCUAnalysis_${arch} PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-deprecated-declarations -Wno-unused-variable -Wno-parentheses -Wno-error=comment>)
target_compile_options(obj.MLIRTritonToGCU_${arch} PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-variable -Wno-deprecated-declarations -Wno-reorder -Wno-unused-but-set-variable -Wno-error=comment>)
target_compile_options(obj.TritonGCUIR_${arch} PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-variable -Wno-error=comment>)
target_compile_options(triton-${arch}-opt PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-variable -Wno-reorder -Wno-error=comment>)
target_compile_options(obj.TritonGCUAnalysis_${arch} PUBLIC $<$<CXX_COMPILER_ID:Clang>:-Wno-sign-compare -Wno-deprecated-declarations -Wno-unused-variable -Wno-parentheses -Wno-error=comment>)

set(KURAMA_TOOLS_TARGET
        triton-${arch}-opt
)

add_custom_target(triton-${arch}-tools ALL DEPENDS
        ${KURAMA_TOOLS_TARGET}
)
