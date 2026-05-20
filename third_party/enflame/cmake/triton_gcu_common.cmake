# Shared between kurama and flagtree (keep this file byte-identical in both trees).
# Include once from triton_gcu.cmake: macros are defined first; bootstrap runs at end of file.
#
# Every macro receives all external data through explicit parameters.
# CMake built-in variables (CMAKE_BINARY_DIR, CMAKE_C_COMPILER, ...) are
# considered ambient context and are NOT passed.

# Nested cmake --build parallelism: honor MAX_JOBS when set, otherwise -j8.
# Sets JOB_SETTING in caller scope (macro, no own scope).
macro(triton_gcu_nested_build_parallelism)
  if(DEFINED ENV{MAX_JOBS} AND NOT "$ENV{MAX_JOBS}" STREQUAL "")
    set(JOB_SETTING "-j$ENV{MAX_JOBS}")
  else()
    set(JOB_SETTING "-j8")
  endif()
  # Ninja: register the job pool used by add_custom_command(... JOB_POOL triton_nested).
  # Use the GLOBAL PROPERTY JOB_POOLS only (CMake Ninja generator reads it; see prop_gbl/JOB_POOLS).
  # Do NOT also set CMAKE_JOB_POOLS here — combining both produced duplicate "pool triton_nested"
  # in rules.ninja.
  #
  # This macro runs once per arch (e.g. gcu300 then gcu400); guard so we APPEND the pool only once.
  get_property(_triton_gcu_nested_pool_registered GLOBAL PROPERTY TRITON_GCU_TRITON_NESTED_POOL_REGISTERED)
  if(NOT _triton_gcu_nested_pool_registered)
    set_property(GLOBAL PROPERTY TRITON_GCU_TRITON_NESTED_POOL_REGISTERED TRUE)
    set_property(GLOBAL APPEND PROPERTY JOB_POOLS "triton_nested=1")
  endif()
endmacro()

# MLIR/LLVM SYSTEM dirs, enflame triton_gcu include dirs, and triton_${_arch}.cmake.
# Parameters:
#   _arch           - architecture tag (gcu300, gcu400, ...)
#   _mlir_inc_dirs  - MLIR include directories (from find_package(MLIR))
#   _llvm_inc_dirs  - LLVM include directories (from find_package(LLVM))
macro(triton_gcu_add_llvm_triton_enflame_base_include_directories _arch _mlir_inc_dirs _llvm_inc_dirs)
  include_directories(SYSTEM ${_mlir_inc_dirs})
  include_directories(SYSTEM ${_llvm_inc_dirs})
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
  include_directories(SYSTEM ${CMAKE_CURRENT_BINARY_DIR}/include)
  include(${CMAKE_CURRENT_LIST_DIR}/triton_${_arch}.cmake)
endmacro()

# Arguments forwarded to the nested Triton CMake configure.
# Parameters:
#   _out_var      - name of the list variable to append to (caller-chosen)
#   _mlir_dir     - MLIR cmake config directory
#   _llvm_lib_dir - LLVM library directory
macro(triton_gcu_append_nested_triton_cmake_args _out_var _mlir_dir _llvm_lib_dir)
  list(APPEND ${_out_var} -DMLIR_DIR=${_mlir_dir})
  list(APPEND ${_out_var} -DLLVM_LIBRARY_DIR=${_llvm_lib_dir})
  list(APPEND ${_out_var} -DTRITON_BUILD_UT=OFF)
  list(APPEND ${_out_var} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER})
  list(APPEND ${_out_var} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})
  list(APPEND ${_out_var} -DCMAKE_BUILD_TYPE=Release)
endmacro()

# In-place patches applied to upstream Triton before the nested configure (shared kurama/flagtree).
# Parameters:
#   triton_src_root - root of the Triton source tree to patch
macro(triton_gcu_apply_common_triton_upstream_patches triton_src_root)
  execute_process(
    COMMAND sed -i "\\#HasParent<\"ModuleOp\">#d"
              ${triton_src_root}/include/triton/Dialect/Triton/IR/TritonOps.td
    ERROR_QUIET
  )
  execute_process(
    COMMAND sed -i "s/\\/\\/ Prevent LLVM's inliner to inline this function/return failure();\\/*/"
              ${triton_src_root}/lib/Conversion/TritonGPUToLLVM/FuncOpToLLVM.cpp
    ERROR_QUIET
  )
  execute_process(
    COMMAND sed -i "s/return success();/*\\//"
              ${triton_src_root}/lib/Conversion/TritonGPUToLLVM/FuncOpToLLVM.cpp
    ERROR_QUIET
  )
  execute_process(
    COMMAND sed -i "\\#nvidia/include/Dialect/NVWS/IR/Dialect.h#d"
              ${triton_src_root}/include/triton/Dialect/TritonGPU/Transforms/Passes.h
    ERROR_QUIET
  )
  execute_process(
    COMMAND sed -i "/nvws::NVWSDialect/d"
              ${triton_src_root}/include/triton/Dialect/TritonGPU/Transforms/Passes.td
    ERROR_QUIET
  )

  execute_process(
      COMMAND sed -i "/maybeLookupNumWarps/,/multiple of 4/ { /multiple of 4/ { N; N; d; }; d; }"
              ${triton_src_root}/lib/Dialect/TritonGPU/IR/Ops.cpp
      ERROR_QUIET
  )
endmacro()

# triton-${_arch}-opt, shared link/compile flags, and triton-${_arch}-tools aggregate target.
# Parameters:
#   _arch       - architecture tag
#   ARGN        - triton object files to link (passed as trailing arguments)
macro(triton_gcu_add_triton_opt_toolchain _arch)
  set(_triton_objs_param ${ARGN})

  set(mlir_register_libs MLIRRegisterAllDialects MLIRRegisterAllExtensions MLIRRegisterAllPasses)
  get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  get_property(mlir_translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
  get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

  add_llvm_executable(triton-${_arch}-opt triton-${_arch}-opt.cpp PARTIAL_SOURCES_INTENDED)
  set_target_properties(triton-${_arch}-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  llvm_update_compile_flags(triton-${_arch}-opt)

  target_link_libraries(triton-${_arch}-opt PRIVATE
    GCUIR_${_arch}
    MemrefExtIR_${_arch}
    MathExtIR_${_arch}
    TritonGCUIR_${_arch}
    MLIRTritonToGCU_${_arch}
    MLIRTritonGCUTransforms_${_arch}
    ${mlir_dialect_libs}
    ${mlir_conversion_libs}
    ${mlir_translation_libs}
    ${mlir_extension_libs}
    ${mlir_register_libs}
    MLIROptLib
    MLIRPass
    MLIRTransforms
    ${_triton_objs_param}
  )

  add_dependencies(triton-${_arch}-opt triton_${_arch})

  mlir_check_all_link_libraries(triton-${_arch}-opt)

  set(_triton_gcu_tools_target triton-${_arch}-opt)
  add_custom_target(triton-${_arch}-tools ALL DEPENDS ${_triton_gcu_tools_target})
endmacro()

# Orchestrator: calls LLVM setup, parallelism, then each stage from triton_gcu.cmake.
# Parameters:
#   _arch           - architecture tag
#   _triton_tag     - git tag/commit for the Triton fetch
#   _git_url_dir    - base git URL directory (e.g. project_git_url_dir)
#   _mlir_dir       - MLIR cmake config directory
#   _llvm_lib_dir   - LLVM library directory
#   _mlir_inc_dirs  - MLIR include directories
#   _llvm_inc_dirs  - LLVM include directories
# Requires triton_${_arch}_objs to be set by the caller (via include of arch .cmake).
macro(triton_gcu_pipeline _arch _triton_tag _git_url_dir _mlir_dir _llvm_lib_dir _mlir_inc_dirs _llvm_inc_dirs)
  include(triton_gcu_llvm)
  include(triton_gcu_llvm_config)
  triton_gcu_nested_build_parallelism()

  triton_gcu_stage_init(${_arch} "${_triton_tag}" "${_git_url_dir}" "${_mlir_inc_dirs}" "${_llvm_inc_dirs}")

  triton_gcu_stage_nested_upstream_triton_build(
    ${_arch} "${_git_url_dir}" "${_mlir_dir}" "${_llvm_lib_dir}"
    "${_triton_build_target}"
    "${third_party_triton_${_arch}_fetch_src}"
    "${third_party_triton_${_arch}_fetch_bin}"
    "${triton_${_arch}_objs}"
    "${JOB_SETTING}"
    "${third_party_triton_${_arch}_src}")
endmacro()


# =============================================================================
# Stage macros (called by triton_gcu_pipeline in triton_gcu_common.cmake)
# =============================================================================

# Fetch Triton source via FetchContent.
# Parameters:
#   _arch           - architecture tag
#   _git_url_dir    - base git URL directory
#   _triton_tag     - full git tag/commit for FetchContent
#   _fetch_bin      - Triton binary (build output) directory
# Sets in caller scope:
#   third_party_triton_${_arch}_fetch_src
#   third_party_triton_${_arch}_src (glob of all source files)
#
# NOTE: Temporary src/bin directories are arch-based (kurama1-compatible) because
# other code depends on these paths.
macro(triton_gcu_fetch_triton_source _arch _git_url_dir _triton_tag _fetch_bin)
  set(third_party_triton_${_arch}_fetch_src "${CMAKE_CURRENT_BINARY_DIR}/third_party_triton_${_arch}_src")

  include(FetchContent)
  FetchContent_Declare(
    third_party_triton_${_arch}_fetch
    GIT_REPOSITORY "${_git_url_dir}/triton.git"
    GIT_TAG "${_triton_tag}"
    SOURCE_DIR "${third_party_triton_${_arch}_fetch_src}"
    BINARY_DIR "${_fetch_bin}"
  )
  FetchContent_GetProperties(third_party_triton_${_arch}_fetch)
  if(NOT third_party_triton_${_arch}_fetch_POPULATED)
    FetchContent_Populate(third_party_triton_${_arch}_fetch)
  endif()

  file(GLOB_RECURSE third_party_triton_${_arch}_src "${third_party_triton_${_arch}_fetch_src}/*")
endmacro()

# Extract Triton version from source tree.
# Parameters:
#   _arch       - architecture tag
#   _src_dir    - Triton source root (containing python/triton/__init__.py)
# Sets in caller scope: TRITON_ORIG_VERSION (+ global property)
macro(triton_gcu_get_triton_version _arch _src_dir)
  set(triton_version_file ${_src_dir}/python/triton/__init__.py)
  execute_process(
    COMMAND grep "__version__ = '" "${triton_version_file}"
    COMMAND sed "s/.*__version__ = '\\([0-9]*\\.[0-9]*\\.[0-9]*\\)'.*/\\1/"
    OUTPUT_VARIABLE _triton_orig_ver_tmp
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _triton_orig_ver_ok)
  if(_triton_orig_ver_tmp AND _triton_orig_ver_ok EQUAL 0)
    set(TRITON_ORIG_VERSION "${_triton_orig_ver_tmp}")
  else()
    set(TRITON_ORIG_VERSION "unknown")
    message(WARNING "[triton-${_arch}] Could not read Triton version from ${triton_version_file}")
  endif()
  message(STATUS "[triton-${_arch}] TRITON_ORIG_VERSION=${TRITON_ORIG_VERSION}")
  set_property(GLOBAL PROPERTY TRITON_ORIG_VERSION "${TRITON_ORIG_VERSION}")
endmacro()

# --- Stages (keep names aligned with flagtree for side-by-side review) ---

# Parameters:
#   _arch           - architecture tag
#   _triton_tag     - git tag/commit for Triton fetch
#   _git_url_dir    - base git URL directory for cloning
#   _mlir_inc_dirs  - MLIR include directories
#   _llvm_inc_dirs  - LLVM include directories
macro(triton_gcu_stage_init _arch _triton_tag _git_url_dir _mlir_inc_dirs _llvm_inc_dirs)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -Wno-unused-parameter")
  triton_distributed_init(${_arch} "${CMAKE_CURRENT_SOURCE_DIR}")

  # Arch-based build target and per-arch src/bin directories (kurama1-compatible).
  set(_triton_build_target "triton_build_${_arch}")
  set(third_party_triton_${_arch}_fetch_bin "${CMAKE_CURRENT_BINARY_DIR}/third_party_triton_${_arch}_bin")

  triton_gcu_fetch_triton_source(
    ${_arch}
    "${_git_url_dir}" "${_triton_tag}"
    "${third_party_triton_${_arch}_fetch_bin}")
  triton_gcu_add_llvm_triton_enflame_base_include_directories(${_arch} "${_mlir_inc_dirs}" "${_llvm_inc_dirs}")
  triton_distributed_add_objs(${_arch})
endmacro()

# Parameters:
#   _arch           - architecture tag
#   _git_url_dir    - base git URL directory (passed down to fetch)
#   _mlir_dir       - MLIR cmake config directory
#   _llvm_lib_dir   - LLVM library directory
#   _build_target   - cmake build target name (from stage_init)
#   _fetch_src      - Triton fetched source directory
#   _fetch_bin      - Triton build output directory
#   _triton_objs    - list of Triton object files to build
#   _job_setting    - parallelism flag (e.g. -j8)
#   _src_files      - glob of all Triton source files (for DEPENDS)
macro(triton_gcu_stage_nested_upstream_triton_build _arch _git_url_dir _mlir_dir _llvm_lib_dir _build_target _fetch_src _fetch_bin _triton_objs _job_setting _src_files)
  triton_distributed_inject(${_arch})
  triton_distributed_configure_upstream_build(${_arch})

  triton_gcu_get_triton_version(${_arch} "${_fetch_src}")
  file(MAKE_DIRECTORY ${_fetch_bin})
  set(_tgnub_cmake_args "")
  triton_gcu_append_nested_triton_cmake_args(_tgnub_cmake_args "${_mlir_dir}" "${_llvm_lib_dir}")
  triton_gcu_apply_common_triton_upstream_patches("${_fetch_src}")

  triton_gcu_nested_upstream_cxx_flags("${_arch}" "-Wno-reorder -Wno-error=comment" _triton_stage_nested_cxx_flags)

  add_custom_command(
    OUTPUT ${_triton_objs}
    COMMAND sed -i "/add_subdirectory\\(test\\)/d" ${_fetch_src}/CMakeLists.txt
    COMMAND sed -i "/add_subdirectory\\(bin\\)/d" ${_fetch_src}/CMakeLists.txt
    COMMAND sed -i "s/-Wno-covered-switch-default//g" ${_fetch_src}/CMakeLists.txt
    COMMAND cmake -S ${_fetch_src} -B ${_fetch_bin} ${_tgnub_cmake_args} -DTRITON_CODEGEN_BACKENDS='nvidia\;amd' "-DCMAKE_CXX_FLAGS=${_triton_stage_nested_cxx_flags}" -G Ninja
    COMMAND cmake --build ${_fetch_bin} --target all ${_job_setting}
    DEPENDS ${_src_files}
    JOB_POOL triton_nested
  )

  add_custom_target(${_build_target} ALL DEPENDS ${_triton_objs})
  message(STATUS "[triton-${_arch}] created build target ${_build_target}")

  add_custom_target(third_party_triton_${_arch}_fetch_build ALL)
  add_dependencies(third_party_triton_${_arch}_fetch_build ${_build_target})

  triton_distributed_add_post_build_copy_libtriton(${_arch})
endmacro()

# Parameters:
#   _arch       - architecture tag
#   _fetch_src  - Triton fetched source directory
#   _fetch_bin  - Triton build output directory
macro(triton_gcu_stage_enflame_subdirectory_and_extra_deps _arch _fetch_src _fetch_bin)
  triton_distributed_add_include_directories(${_arch})

  foreach(_kurama_triton_obj IN ITEMS
      obj.MLIRGCUTritonToTritonGPU_${_arch}
      obj.MLIRTritonToGCU_${_arch}
      obj.MLIRTritonGCUTransforms_${_arch}
      obj.TritonGCUAnalysis_${_arch}
      obj.TritonGCUIR_${_arch}
      obj.GCUWSIR_${_arch}
    )
    if(TARGET ${_kurama_triton_obj})
      add_dependencies(${_kurama_triton_obj} third_party_triton_${_arch}_fetch_build)
    endif()
  endforeach()

  set(_gcu_compiler_tablegen_targets
      GCUTableGen_ MemrefExtTableGen_ MathExtTableGen_
      LIVENESSTableGen MLIRGCUPassIncGen MLIRGCUConversionPassIncGen)
  foreach(_kurama_gcu_obj IN ITEMS
      obj.GCUIR_${_arch}
      obj.MemrefExtIR_${_arch}
      obj.MathExtIR_${_arch}
      obj.MLIRTritonToGCU_${_arch}
      obj.MLIRTritonGCUTransforms_${_arch}
    )
    if(TARGET ${_kurama_gcu_obj})
      foreach(_tg IN LISTS _gcu_compiler_tablegen_targets)
        if(TARGET ${_tg})
          add_dependencies(${_kurama_gcu_obj} ${_tg})
        endif()
      endforeach()
    endif()
  endforeach()
endmacro()

# Parameters:
#   _arch        - architecture tag
#   _triton_objs - Triton object files to link
macro(triton_gcu_stage_triton_opt_link_and_compile _arch _triton_objs)
  triton_gcu_add_triton_opt_toolchain(${_arch} ${_triton_objs})

  target_link_libraries(triton-${_arch}-opt PRIVATE MLIRVectorToGCU)

  if(TARGET MLIRGCUTritonToTritonGPU_${_arch})
    target_link_libraries(triton-${_arch}-opt PRIVATE MLIRGCUTritonToTritonGPU_${_arch})
  endif()
endmacro()

macro(triton_gcu_stage_unittests)
  set(_triton_gcu_unittests ${CMAKE_CURRENT_SOURCE_DIR}/../unittests/CMakeLists.txt)
  if(EXISTS ${_triton_gcu_unittests})
    include(${_triton_gcu_unittests})
  endif()
endmacro()

# =============================================================================
# Function-based APIs (used by triton_gcu{300,400,500}/CMakeLists.txt)
# =============================================================================

# Nested upstream Triton uses a separate CMake invocation; it does not inherit directory
# COMPILE_DEFINITIONS from Kurama. Pass -D when distributed is enabled for gcu400.
function(triton_gcu_nested_upstream_cxx_flags _arch _base_flags _out_var)
  set(_flags "${_base_flags}")
  if(ENABLE_TRITON_DISTRIBUTED AND "${_arch}" STREQUAL "gcu400")
    set(_flags "${_base_flags} -DENABLE_TRITON_DISTRIBUTED")
  endif()
  set(${_out_var} "${_flags}" PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# Function: build_triton_python_bindings
# -----------------------------------------------------------------------------
# Builds pybind11 Python bindings for multiple Python versions.
#
# Creates _triton_${arch}.cpython-3X.so for each detected Python version.
#
# Parameters:
#   ARCH_NAME  : Architecture name (gcu300, gcu400, gcu500)
#   SOURCE_DIR : Directory containing triton_${arch}_module.cpp
#
function(build_triton_python_bindings ARCH_NAME SOURCE_DIR)
  string(TOUPPER "${ARCH_NAME}" _ARCH_UPPER)
  set(_ARCH_TAG "${ARCH_NAME}")

  if(NOT DEFINED ${_ARCH_UPPER}_PYTHON_VERSIONS)
    set(_CANDIDATE_VERSIONS "3.9;3.10;3.11;3.12")
    if(DEFINED ENV{PYTHON_VERSION} AND NOT "$ENV{PYTHON_VERSION}" STREQUAL "")
      list(APPEND _CANDIDATE_VERSIONS "$ENV{PYTHON_VERSION}")
      list(REMOVE_DUPLICATES _CANDIDATE_VERSIONS)
    endif()
    set(${_ARCH_UPPER}_PYTHON_VERSIONS "")
    foreach(_cv IN LISTS _CANDIDATE_VERSIONS)
      unset(_cv_exe)
      unset(_cv_exe CACHE)
      find_program(_cv_exe "python${_cv}" NO_CACHE)
      if(_cv_exe)
        list(APPEND ${_ARCH_UPPER}_PYTHON_VERSIONS "${_cv}")
      endif()
    endforeach()
    if(NOT ${_ARCH_UPPER}_PYTHON_VERSIONS)
      set(${_ARCH_UPPER}_PYTHON_VERSIONS "3.10")
    endif()
  endif()
  message(STATUS "[${_ARCH_TAG}-python] Building bindings for: ${${_ARCH_UPPER}_PYTHON_VERSIONS}")

  set(_BINDING_TARGETS "")

  foreach(_pyver IN LISTS ${_ARCH_UPPER}_PYTHON_VERSIONS)
    unset(_PY_EXE)
    unset(_PY_EXE CACHE)
    find_program(_PY_EXE "python${_pyver}" NO_CACHE)
    if(NOT _PY_EXE)
      message(WARNING "[${_ARCH_TAG}-python] python${_pyver} not found -- skipping")
      continue()
    endif()

    execute_process(COMMAND "${_PY_EXE}" -c
      "import pybind11; print(pybind11.get_include())"
      OUTPUT_VARIABLE _pb_inc OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET RESULT_VARIABLE _pb_rc)
    if(NOT _pb_rc EQUAL 0)
      message(WARNING "[${_ARCH_TAG}-python] pybind11 not available for python${_pyver} -- installing")
      execute_process(COMMAND "${_PY_EXE}" -m pip install --user pybind11)
      execute_process(COMMAND "${_PY_EXE}" -c
        "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE _pb_inc OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()

    execute_process(COMMAND "${_PY_EXE}" -c
      "import sysconfig; print(sysconfig.get_path('include'))"
      OUTPUT_VARIABLE _py_inc OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(COMMAND "${_PY_EXE}" -c
      "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
      OUTPUT_VARIABLE _py_ext_suffix OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE "." "" _py_tag "${_pyver}")

    set(_tgt "_triton_${_ARCH_TAG}_py${_py_tag}")

    add_library(${_tgt} MODULE "${SOURCE_DIR}/triton_${_ARCH_TAG}_module.cpp")
    target_compile_features(${_tgt} PRIVATE cxx_std_17)
    target_include_directories(${_tgt} PRIVATE
      "${SOURCE_DIR}"
      "${CMAKE_CURRENT_SOURCE_DIR}/lib"
    )
    target_include_directories(${_tgt} SYSTEM PUBLIC
      "${_pb_inc}" "${_py_inc}"
    )
    target_link_libraries(${_tgt} PRIVATE triton_${_ARCH_TAG}_core)

    set_target_properties(${_tgt} PROPERTIES
      OUTPUT_NAME "_triton_${_ARCH_TAG}"
      PREFIX ""
      SUFFIX "${_py_ext_suffix}"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
      BUILD_RPATH "$ORIGIN"
      INSTALL_RPATH "$ORIGIN"
      LINK_FLAGS "-Wl,--version-script=${SOURCE_DIR}/triton_${_ARCH_TAG}_py.map"
    )

    list(APPEND _BINDING_TARGETS "${_tgt}")
    message(STATUS "[${_ARCH_TAG}-python]   ${_pyver} -> ${_tgt} (${_PY_EXE})")
  endforeach()

  if(_BINDING_TARGETS)
    add_custom_target(_triton_${_ARCH_TAG} ALL DEPENDS ${_BINDING_TARGETS})
  endif()
endfunction()
