# triton_gcu/unittests/CMakeLists.txt
#
# Unit tests for triton_gcu.
# This file is included (not add_subdirectory'd) per-arch from
# cmake/triton_gcu.cmake, so ${arch}, ${ARCH}, MLIR include paths,
# and all library targets are available in the caller's scope.
#
# We use add_executable + system gtest instead of LLVM's add_unittest
# because the prebuild LLVM does not ship llvm_gtest / llvm_gtest_main.

if(NOT TARGET TritonGCUUnitTests)
  add_custom_target(TritonGCUUnitTests)
  set_target_properties(TritonGCUUnitTests PROPERTIES
    FOLDER "TritonGCU Unit Tests")
endif()

set(_unittest_dir ${CMAKE_CURRENT_LIST_DIR})

set(_conversion_common_dir ${triton_gcu_source_dir}/lib/Conversion/Common)

# Resolve GTest: both static archives live in the same directory. Prefer CMake
# targets; otherwise pick the first directory that contains both libgtest.a and
# libgtest_main.a (avoids hard-coded /usr/local/lib and Ninja "missing file" errors).
if(NOT _triton_gcu_gtest_link_libs)
  find_package(GTest CONFIG QUIET)
  if(NOT GTest_FOUND)
    find_package(GTest MODULE QUIET)
  endif()
  if(GTest_FOUND AND TARGET GTest::gtest_main AND TARGET GTest::gtest)
    set(_triton_gcu_gtest_link_libs GTest::gtest_main GTest::gtest)
  else()
    foreach(_d IN ITEMS
        "$ENV{GTEST_ROOT}/lib"
        "/usr/local/lib"
        "/usr/local/lib64"
        "/usr/lib"
        "/usr/lib64"
      )
      if(EXISTS "${_d}/libgtest.a" AND EXISTS "${_d}/libgtest_main.a")
        set(_triton_gcu_gtest_link_libs
          "${_d}/libgtest_main.a"
          "${_d}/libgtest.a")
        break()
      endif()
    endforeach()
  endif()
endif()
if(NOT _triton_gcu_gtest_link_libs)
  message(WARNING
    "triton_gcu unittests: GTest not found. Install googletest / libgtest-dev, "
    "or set GTEST_ROOT, or add CMAKE_PREFIX_PATH so that find_package(GTest) works.")
  return()
endif()

# Helper: register a gtest executable linked against system gtest.
macro(_add_triton_gcu_unittest _umbrella _name)
  add_executable(${_name} ${ARGN})
  target_link_libraries(${_name} PRIVATE ${_triton_gcu_gtest_link_libs} pthread)
  add_dependencies(${_umbrella} ${_name})
endmacro()

set(_unittest_common_libs
  -Wl,--start-group
  MLIRTritonToGCU_${arch}
  MLIRTritonGCUTransforms_${arch}
  TritonGCUIR_${arch}
  GCUIR_${arch}
  MemrefExtIR_${arch}
  MathExtIR_${arch}
  MLIRPass
  MLIRTransforms
  -Wl,--end-group
)
# ---------------------------------------------------------------------------
# Public function: register a triton_gcu unit test.
#
# Usage:
#   add_triton_gcu_test(<TestName> <source_relative_to_unittest_dir>)
#
# Example:
#   add_triton_gcu_test(DoMemsetTest Conversion/TritonToGCU/DoMemsetTest.cpp)
#
# The function creates target <TestName>_${arch}, links common libs, sets
# arch compile definitions, and installs into the c_kurama_test component.
# The directory part of _src is preserved in both the build output and
# install tree, e.g. Conversion/TritonToGCU/Foo.cpp produces:
#   build:   ${CMAKE_BINARY_DIR}/unittests/Conversion/TritonToGCU/Foo_${arch}
#   install: <prefix>/unittests/Conversion/TritonToGCU/Foo_${arch}
# ---------------------------------------------------------------------------
function(add_triton_gcu_test _name _src)
  set(_target ${_name}_${arch})
  get_filename_component(_subdir ${_src} DIRECTORY)

  set_source_files_properties(${TRITON_OUTPUT_FILES} PROPERTIES
    EXTERNAL_OBJECT TRUE GENERATED TRUE)

  _add_triton_gcu_unittest(TritonGCUUnitTests ${_target}
    ${_unittest_dir}/${_src}
    ${TRITON_OUTPUT_FILES}
  )

  target_compile_definitions(${_target} PRIVATE
    $<$<STREQUAL:${arch},gcu400>:TEST_GCU400>
    $<$<STREQUAL:${arch},gcu300>:TEST_GCU300>
  )

  target_link_libraries(${_target} PRIVATE
    triton_${arch}_core
    ${_unittest_common_libs}
    ${TRITON_MLIR_DEPS}
  )

  if(TARGET ${TRITON_BUILD_TARGET})
    add_dependencies(${_target} ${TRITON_BUILD_TARGET})
  endif()

  if(arch STREQUAL "gcu400")
    target_link_libraries(${_target} PRIVATE GCUWSIR_${arch})
  endif()

  target_include_directories(${_target} PRIVATE ${_conversion_common_dir})
  set_target_properties(${_target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unittests/${_subdir})

  install(TARGETS ${_target} DESTINATION unittests/${_subdir} COMPONENT c_kurama_test)
endfunction()

# ---------- Conversion/TritonToGCU tests ----------
add_triton_gcu_test(DoMemsetTest             Conversion/TritonToGCU/DoMemsetTest.cpp)
add_triton_gcu_test(MergeContinuousDimsTest  Conversion/TritonToGCU/MergeContinuousDimsTest.cpp)
add_triton_gcu_test(CreatePrintfOpTest       Conversion/TritonToGCU/CreatePrintfOpTest.cpp)
add_triton_gcu_test(ConfigGcuLoadExTest      Conversion/TritonToGCU/ConfigGcuLoadExTest.cpp)
add_triton_gcu_test(ConfigGcuLoadTest        Conversion/TritonToGCU/ConfigGcuLoadTest.cpp)
add_triton_gcu_test(TmaCopyBarrierTest       Conversion/TritonToGCU/TmaCopyBarrierTest.cpp)
add_triton_gcu_test(CopyFromSharedMemTest   Conversion/TritonToGCU/CopyFromSharedMemTest.cpp)
add_triton_gcu_test(StoreToSharedMemTest    Conversion/TritonToGCU/StoreToSharedMemTest.cpp)
add_triton_gcu_test(PtrAnalysisTest         Conversion/TritonToGCU/PtrAnalysisTest.cpp)
# PtrAnalysisTest creates MLIR objects with virtual methods (IRRewriter).
# MLIR is built with -fno-rtti, so the test must match to avoid typeinfo
# link errors.  Basic gtest macros (TEST_F, EXPECT_*) work without RTTI.
target_compile_options(PtrAnalysisTest_${arch} PRIVATE -fno-rtti)
add_triton_gcu_test(ReduceScanCommonTest    Conversion/TritonToGCU/ReduceScanCommonTest.cpp)
target_compile_options(ReduceScanCommonTest_${arch} PRIVATE -fno-rtti)
add_triton_gcu_test(VectorizeTest           Conversion/TritonToGCU/VectorizeTest.cpp)
target_compile_options(VectorizeTest_${arch} PRIVATE -fno-rtti)
add_triton_gcu_test(ConvertTensorPointerTest Conversion/TritonToGCU/ConvertTensorPointerTest.cpp)
target_compile_options(ConvertTensorPointerTest_${arch} PRIVATE -fno-rtti)
add_triton_gcu_test(TritonToGCULayoutTest    Conversion/TritonToGCU/TritonToGCULayoutTest.cpp)
target_compile_options(TritonToGCULayoutTest_${arch} PRIVATE -fno-rtti)

# ---------- Analysis tests ----------
add_triton_gcu_test(AxisInfoExTest              Analysis/AxisInfoExTest.cpp)
target_compile_options(AxisInfoExTest_${arch} PRIVATE -fno-rtti)
add_triton_gcu_test(OpFoldResultUtilsTest       Analysis/OpFoldResultUtilsTest.cpp)
target_compile_options(OpFoldResultUtilsTest_${arch} PRIVATE -fno-rtti)

# ---------- Conversion/Common tests (gcu300 only) ----------
if(arch STREQUAL "gcu300")
  add_triton_gcu_test(ConstantUtilTest            Conversion/Common/ConstantUtilTest.cpp)
endif()

# ---------- Transforms/WarpSpecialization tests (gcu400 only) ----------
if(arch STREQUAL "gcu400")
  add_triton_gcu_test(CodePartitionUtilityTest    Transforms/WarpSpecialization/CodePartitionUtilityTest.cpp)
  target_compile_options(CodePartitionUtilityTest_${arch} PRIVATE -fno-rtti)
endif()
