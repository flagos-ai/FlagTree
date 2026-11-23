# Build against the same LLVM as the intended opt or clang. FlagTree also
# links KL3GroupedExp.cpp directly into its in-process LLVM pipeline.
cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(KL3GroupedExpPlugin LANGUAGES C CXX)
  find_package(LLVM REQUIRED CONFIG)
endif()

add_library(KL3GroupedExp MODULE KL3GroupedExp.cpp PassPlugin.cpp)
target_include_directories(KL3GroupedExp PRIVATE
  ${LLVM_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../../../../include)
target_compile_features(KL3GroupedExp PRIVATE cxx_std_17)
if(NOT LLVM_ENABLE_RTTI AND NOT MSVC)
  target_compile_options(KL3GroupedExp PRIVATE -fno-rtti)
endif()
target_link_libraries(KL3GroupedExp PRIVATE "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>")
set_target_properties(KL3GroupedExp PROPERTIES PREFIX "")
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set_target_properties(KL3GroupedExp PROPERTIES EXCLUDE_FROM_ALL ON)
endif()

set(KL3_OPT "${LLVM_TOOLS_BINARY_DIR}/opt" CACHE FILEPATH "XPU-enabled opt matching this LLVM")
set(fixture "${CMAKE_CURRENT_SOURCE_DIR}/../../../../test/kl3-grouped-exp.ll")
add_custom_target(check-kl3-grouped-exp
  COMMAND "${KL3_OPT}" -load-pass-plugin=$<TARGET_FILE:KL3GroupedExp>
          -passes=kl3-grouped-exp,kl3-grouped-exp,verify -S "${fixture}" -o kl3-exp.ll
  COMMAND "${LLVM_TOOLS_BINARY_DIR}/FileCheck" "${fixture}" --input-file=kl3-exp.ll
  COMMAND "${KL3_OPT}" -mtriple=xpu2-baidu-none-gnu -load-pass-plugin=$<TARGET_FILE:KL3GroupedExp>
          -passes=kl3-grouped-exp,verify -S "${fixture}" -o kl2-exp.ll
  COMMAND "${LLVM_TOOLS_BINARY_DIR}/FileCheck" "${fixture}" --input-file=kl2-exp.ll
          --check-prefix=KL2 "--implicit-check-not=asm sideeffect"
  DEPENDS KL3GroupedExp VERBATIM)
