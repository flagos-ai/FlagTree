# Flagtree builds Triton once in the top-level CMake graph. Enflame links those
# targets directly, so backend source overrides under spec_cpp are applied by
# FlagTreeBackendSpec.cmake to the same targets consumed here.
include_guard(GLOBAL)

function(setup_triton_in_tree)
  set(TRITON_SOURCE_DIR ${CMAKE_SOURCE_DIR} PARENT_SCOPE)
  set(TRITON_BINARY_DIR ${CMAKE_BINARY_DIR} PARENT_SCOPE)
  set(TRITON_ORIG_VERSION "${TRITON_VERSION}" PARENT_SCOPE)
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
  set(TRITON_CORE_TABLEGEN_TARGETS
    TritonTableGen
    TritonGPUTableGen
    TritonGPUAttrDefsIncGen
    TritonGPUCTAAttrIncGen
    TritonGPUTypeInterfacesIncGen
    TritonGPUOpInterfacesIncGen
    TritonGPUTransformsIncGen
    TritonConversionPassIncGen
    TritonTransformsIncGen
    TritonNvidiaGPUTableGen
    TritonNvidiaGPUAttrDefsIncGen
    TritonNvidiaGPUOpInterfacesIncGen
    TritonNvidiaGPUTransformsIncGen
    PARENT_SCOPE
  )
endfunction()

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
