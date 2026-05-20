# For Triton
#
# Staged pipeline matches kurama/cmake/triton_gcu.cmake:
# same macro names and call order; implementations differ (FetchContent vs in-tree).

include(triton_gcu_common)
include(triton_gcu_llvm)

macro(triton_gcu_local_triton_source)
  set(third_party_triton_${arch}_fetch_src "${CMAKE_SOURCE_DIR}")
  file(GLOB_RECURSE third_party_triton_${arch}_src "${CMAKE_SOURCE_DIR}/include/*" "${CMAKE_SOURCE_DIR}/lib/*" "${CMAKE_SOURCE_DIR}/third_party/f2reduce/*" "${CMAKE_SOURCE_DIR}/third_party/proton/*")
endmacro()

# --- Stages (keep names aligned with kurama for side-by-side review) ---

macro(triton_gcu_stage_init)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-but-set-parameter -Wno-attributes")
  set(_triton_build_target triton_build_in_flagtree)
  set(third_party_triton_${arch}_fetch_bin "${CMAKE_BINARY_DIR}/triton_bin")
  triton_gcu_local_triton_source()

  triton_gcu_add_llvm_triton_enflame_base_include_directories(${arch} "${MLIR_INCLUDE_DIRS}" "${LLVM_INCLUDE_DIRS}")

  # In flagtree Triton is built in-tree: CMAKE_SOURCE_DIR IS the Triton root.
  # All variables that setup_triton_fetch would set via PARENT_SCOPE are defined
  # here instead (macros share the caller's scope, avoiding PARENT_SCOPE issues).
  set(TRITON_SOURCE_DIR ${CMAKE_SOURCE_DIR})
  set(TRITON_BINARY_DIR ${CMAKE_BINARY_DIR})
  set(TRITON_BUILD_TARGET "triton_build_in_flagtree")
  set(TRITON_OUTPUT_FILES "")
  set(TRITON_CORE_LIBS
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonTransforms
    TritonToTritonGPU
    TritonAnalysis
    TritonGPUToLLVM
    TritonLLVMIR
    TritonTools
    GluonIR
    GluonTransforms
  )
  include_directories(SYSTEM ${TRITON_SOURCE_DIR}/include)
  include_directories(SYSTEM ${TRITON_BINARY_DIR}/include)
endmacro()

macro(triton_gcu_stage_nested_upstream_triton_build)
  if(TARGET ${_triton_build_target})
    message(STATUS "[triton-${arch}] reuse ${_triton_build_target}")
  else()
    file(MAKE_DIRECTORY ${third_party_triton_${arch}_fetch_bin})

    triton_gcu_append_nested_triton_cmake_args(triton_cmake_args "${MLIR_DIR}" "${LLVM_LIBRARY_DIR}")
    triton_gcu_apply_common_triton_upstream_patches("${third_party_triton_${arch}_fetch_src}")

    execute_process(
      COMMAND sed -i "/auto dTy = cast<ShapedType>(\\$_op.getD().getType());/d"
                ${third_party_triton_${arch}_fetch_src}/include/triton/Dialect/Triton/IR/TritonOpInterfaces.td
      ERROR_QUIET
    )

    add_custom_command(
      OUTPUT ${triton_${arch}_objs}
      COMMAND sed -i "s/-Wno-covered-switch-default//g" ${third_party_triton_${arch}_fetch_src}/CMakeLists.txt
      COMMAND find ${third_party_triton_${arch}_fetch_src} -name "CMakeLists.txt" -exec sed -i "s/-Wno-covered-switch-default//g" {} +
      COMMAND cmake -S ${third_party_triton_${arch}_fetch_src} -B ${third_party_triton_${arch}_fetch_bin} ${triton_cmake_args} -DTRITON_CODEGEN_BACKENDS='nvidia\;amd' -DCMAKE_CXX_FLAGS='-Wno-reorder -Wno-error=comment -Wno-unknown-warning-option' -G Ninja
      COMMAND cmake --build ${third_party_triton_${arch}_fetch_bin} --target all ${JOB_SETTING}
      DEPENDS ${third_party_triton_${arch}_src}
    )

    add_custom_target(${_triton_build_target} ALL DEPENDS ${triton_${arch}_objs})
    message(STATUS "[triton-${arch}] created ${_triton_build_target}")
  endif()

  add_custom_target(third_party_triton_${arch}_fetch_build ALL)
  add_dependencies(third_party_triton_${arch}_fetch_build ${_triton_build_target})
endmacro()

macro(triton_gcu_stage_enflame_subdirectory_and_extra_deps)
  triton_gcu_add_triton_enflame_subdirectory_bundle(${arch} "${third_party_triton_${arch}_fetch_src}" "${third_party_triton_${arch}_fetch_bin}")
endmacro()

macro(triton_gcu_stage_triton_opt_link_and_compile)
  triton_gcu_add_triton_opt_toolchain(${arch})

  target_link_options(triton-${arch}-opt PRIVATE -Wl,--allow-multiple-definition)

  target_link_libraries(triton-${arch}-opt PRIVATE
    TleIR
    TleToLLVM
    TritonTLETransforms
    TritonGPUTransforms
  )
endmacro()

macro(triton_gcu_stage_unittests)
  # Kurama registers gtests here; flagtree has no triton_gcu unit tests in-tree.
endmacro()


# -----------------------------------------------------------------------------
# Function: setup_triton_fetch
# -----------------------------------------------------------------------------
# Sets up Triton source fetch and build configuration (organized by commit).
# If the same commit was already fetched and built, this function skips the work.
#
# Parameters:
#   TRITON_COMMIT     : Triton commit hash/tag
#   MLIR_DIR          : MLIR cmake directory
#   LLVM_LIBRARY_DIR  : LLVM library directory
#   OBJECT_FILES      : List of Triton object files to build
#   GIT_URL_DIR       : Base git URL directory for cloning triton.git
#   MLIR_TABLEGEN_EXE : Path to mlir-tblgen executable
#
# Exports to PARENT_SCOPE:
#   TRITON_BUILD_TARGET : Name of the build target
#   TRITON_SOURCE_DIR   : Triton source directory
#   TRITON_BINARY_DIR   : Triton build directory
#   TRITON_OUTPUT_FILES : Triton object files (absolute paths)
#   TRITON_OBJECT_LIB   : CMake OBJECT library target name (nested Triton objs; triton_upstream_objs_<arch>)
#   TRITON_ORIG_VERSION : Extracted Triton version
#
function(setup_triton_fetch)
  cmake_parse_arguments(ARG "" "TRITON_COMMIT;MLIR_DIR;LLVM_LIBRARY_DIR;GIT_URL_DIR;MLIR_TABLEGEN_EXE" "OBJECT_FILES" ${ARGN})
  set(TRITON_SOURCE_DIR ${CMAKE_SOURCE_DIR} PARENT_SCOPE)
  set(TRITON_BINARY_DIR ${CMAKE_BINARY_DIR} PARENT_SCOPE)
  set(TRITON_BUILD_TARGET "triton_build_in_flagtree" PARENT_SCOPE)
  # In flagtree, Triton is built in-tree as OBJECT libraries rather than
  # via nested build .o files.  TRITON_OUTPUT_FILES is intentionally empty;
  # TRITON_CORE_LIBS carries the in-tree OBJECT library targets that the
  # core shared library must link against.
  set(TRITON_OUTPUT_FILES "" PARENT_SCOPE)
  set(TRITON_CORE_LIBS
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonTransforms
    TritonToTritonGPU
    TritonAnalysis
    TritonGPUToLLVM
    TritonLLVMIR
    TritonTools
    GluonIR
    GluonTransforms
    PARENT_SCOPE
  )
endfunction()

triton_gcu_pipeline(${arch} 0 "${project_git_url_dir}" "${MLIR_DIR}" "${LLVM_LIBRARY_DIR}" "${MLIR_INCLUDE_DIRS}" "${LLVM_INCLUDE_DIRS}")
