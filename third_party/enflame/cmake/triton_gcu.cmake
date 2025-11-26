# For Triton

# ######################################################
# Get LLVM for triton
include(triton_gcu_llvm)
include(triton_gcu_llvm_config)

# Disable warnings that show up in external code (gtest;pybind11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-but-set-parameter")
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include) # Tablegen'd files

# 使用本地的 triton 文件，不需要下载
set(third_party_triton_${arch}_fetch_src "${CMAKE_CURRENT_LIST_DIR}/../include/triton")
set(third_party_triton_${arch}_fetch_bin "${CMAKE_CURRENT_BINARY_DIR}/third_party_triton_${arch}_bin")
file(GLOB_RECURSE third_party_triton_${arch}_src "${CMAKE_CURRENT_LIST_DIR}/../include/triton/*")

set(TRITON_SOURCE_DIR ${third_party_triton_${arch}_fetch_src})
message(STATUS "TRITON_SOURCE_DIR: ${TRITON_SOURCE_DIR}")
set(TRITON_VERSION_FILE ${third_party_triton_${arch}_fetch_src}/python/triton/__init__.py)

# 提取版本号
execute_process(
    COMMAND grep "__version__ = '" ${TRITON_VERSION_FILE}
    COMMAND sed "s/.*__version__ = '\\([0-9]*\\.[0-9]*\\.[0-9]*\\)'.*/\\1/"
    OUTPUT_VARIABLE TRITON_ORIG_VERSION_TEMP
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE VERSION_EXTRACT_RESULT
)

if(TRITON_ORIG_VERSION_TEMP AND VERSION_EXTRACT_RESULT EQUAL 0)
    message(STATUS "Successfully extracted Triton version: ${TRITON_ORIG_VERSION_TEMP}")
    set(TRITON_ORIG_VERSION ${TRITON_ORIG_VERSION_TEMP} CACHE STRING "Triton original version" FORCE)
else()
    message(WARNING "Could not extract version from ${TRITON_VERSION_FILE}")
    set(TRITON_ORIG_VERSION "unknown" CACHE STRING "Triton original version" FORCE)
endif()

include(${CMAKE_CURRENT_LIST_DIR}/triton_${arch}.cmake)

file(MAKE_DIRECTORY ${third_party_triton_${arch}_fetch_bin})

list(APPEND triton_cmake_args -DMLIR_DIR=${MLIR_DIR})
list(APPEND triton_cmake_args -DLLVM_LIBRARY_DIR=${LLVM_LIBRARY_DIR})
list(APPEND triton_cmake_args -DTRITON_BUILD_UT=OFF)
list(APPEND triton_cmake_args -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
list(APPEND triton_cmake_args -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
list(APPEND triton_cmake_args -DCMAKE_BUILD_TYPE=Release)

add_custom_command(
  OUTPUT  ${triton_${arch}_objs}
  COMMAND sed -i "/add_subdirectory\\(test\\)/d" ${third_party_triton_${arch}_fetch_src}/CMakeLists.txt
  COMMAND sed -i "/add_subdirectory\\(bin\\)/d" ${third_party_triton_${arch}_fetch_src}/CMakeLists.txt
  COMMAND cmake -S ${third_party_triton_${arch}_fetch_src} -B ${third_party_triton_${arch}_fetch_bin} ${triton_cmake_args} -DCMAKE_CXX_FLAGS="-Wno-reorder" -G Ninja
  COMMAND cmake --build ${third_party_triton_${arch}_fetch_bin} --target all ${JOB_SETTING}
  DEPENDS ${third_party_triton_${arch}_src}
)

add_custom_target(third_party_triton_${arch}_fetch_build ALL DEPENDS ${triton_${arch}_objs})

add_library(triton_${arch} INTERFACE)
add_dependencies(triton_${arch} third_party_triton_${arch}_fetch_build)

message(STATUS "third_party_triton_${arch}_fetch_bin is ${third_party_triton_${arch}_fetch_bin}")


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
add_subdirectory(test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)


add_llvm_executable(triton-${arch}-opt triton-${arch}-opt.cpp PARTIAL_SOURCES_INTENDED)
set_target_properties(triton-${arch}-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

llvm_update_compile_flags(triton-${arch}-opt)

target_link_libraries(triton-${arch}-opt PRIVATE
GCUIR${arch}
MemrefExtIR${arch}
MathExtIR${arch}
TritonGCUIR_${arch}
TritonGCUTestAnalysis_${arch}
MLIRTritonToGCU_${arch}
MLIRTritonGCUTransforms_${arch}
${dialect_libs}
${conversion_libs}
${translation_libs}
${extension_libs}
# MLIR core
MLIROptLib
MLIRPass
MLIRTransforms
${triton_${arch}_objs}
)
add_dependencies(triton-${arch}-opt triton_${arch})

mlir_check_all_link_libraries(triton-${arch}-opt)

# target_compile_options(obj.TritonGCUAnalysis_${arch} PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare>)
# target_compile_options(obj.MLIRTritonToGCU_${arch} PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-variable>)
# target_compile_options(obj.TritonGCUIR_${arch} PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-variable>)
# target_compile_options(obj.MLIRMemRefToGCU PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-variable>)
# target_compile_options(triton-${arch}-opt PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-variable>)
###

# target_compile_options(TritonGPUToLLVM PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized -Wno-extra -Wno-unused-variable>)
# target_compile_options(TritonGPUTransforms PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized -Wno-extra>)
# target_compile_options(TritonIR PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-but-set-variable -Wno-unused-variable>)
# target_compile_options(TritonGPUIR PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized -Wno-extra -Wno-reorder -Wno-parentheses>)
# target_compile_options(TritonTransforms PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized -Wno-extra>)
# target_compile_options(TritonTools PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-but-set-variable -Wno-unused-function -Wno-unused-variable -Wno-parentheses>)
# target_compile_options(TritonNvidiaGPUIR PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-comment -Wno-reorder>)
# target_compile_options(PrintLoadStoreMemSpaces PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-variable>)
# target_compile_options(TritonLLVMIR PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-unused-variable -Wno-unused-but-set-variable>)
target_compile_options(obj.TritonGCUAnalysis_${arch} PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-deprecated-declarations -Wno-unused-variable -Wno-parentheses -Wno-comment -Wno-maybe-uninitialized>)
target_compile_options(obj.MLIRTritonToGCU_${arch} PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-variable -Wno-deprecated-declarations -Wno-reorder -Wno-unused-but-set-variable -Wno-comment -Wno-maybe-uninitialized>)
target_compile_options(obj.TritonGCUIR_${arch} PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-variable -Wno-comment -Wno-maybe-uninitialized>)

target_compile_options(triton-${arch}-opt PUBLIC $<$<CXX_COMPILER_ID:GNU>:-Wno-sign-compare -Wno-unused-variable -Wno-reorder -Wno-maybe-uninitialized>)

# target_compile_options(TritonIR PUBLIC $<$<CXX_COMPILER_ID:Clang>:-Wno-sign-compare -Wno-unused-but-set-variable -Wno-unused-variable>)
# target_compile_options(TritonGPUIR PUBLIC $<$<CXX_COMPILER_ID:Clang>:-Wno-extra -Wno-reorder -Wno-parentheses -Wno-unused-function>)
# target_compile_options(TritonAnalysis PUBLIC $<$<CXX_COMPILER_ID:Clang>:-Wno-reorder-ctor>)
target_compile_options(obj.TritonGCUAnalysis_${arch} PUBLIC $<$<CXX_COMPILER_ID:Clang>:-Wno-sign-compare -Wno-deprecated-declarations -Wno-unused-variable -Wno-parentheses -Wno-comment -Wno-maybe-uninitialized>)
# target_compile_options(TritonTransforms PUBLIC $<$<CXX_COMPILER_ID:Clang>:-Wno-deprecated-copy>)
# target_compile_options(TritonTools PUBLIC $<$<CXX_COMPILER_ID:Clang>:-Wno-sign-compare -Wno-unused-function -Wno-unused-variable -Wno-parentheses>)

set(KURAMA_TOOLS_TARGET
        triton-${arch}-opt
)

add_custom_target(triton-${arch}-tools ALL DEPENDS
        ${KURAMA_TOOLS_TARGET}
)

# 将TRITON_ORIG_VERSION变量提升到所有上级作用域
set(TRITON_ORIG_VERSION ${TRITON_ORIG_VERSION} PARENT_SCOPE)
set(TRITON_ORIG_VERSION ${TRITON_ORIG_VERSION} CACHE STRING "Triton original version" FORCE)
set_property(GLOBAL PROPERTY TRITON_ORIG_VERSION ${TRITON_ORIG_VERSION})

message(STATUS "TRITON_ORIG_VERSION (${TRITON_ORIG_VERSION}) is now available in all parent scopes")
