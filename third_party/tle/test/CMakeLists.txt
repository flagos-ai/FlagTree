
llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
)

# Tool targets are configured after the TLE plugin. Resolve their availability
# at the end of the root directory, including builds with C++ unit tests off.
function(tle_add_lit_tests)
  set(TRITON_TLE_TEST_DEPENDS
    triton-opt
    triton-tensor-layout
    triton-llvm-opt
  )
  foreach(tool IN LISTS TRITON_TLE_TEST_DEPENDS)
    if(NOT TARGET ${tool})
      return()
    endif()
  endforeach()

  get_directory_property(tle_test_binary_dir
    DIRECTORY "${CMAKE_CURRENT_FUNCTION_LIST_DIR}" BINARY_DIR)
  set(FILECHECK_PATH "${LLVM_LIBRARY_DIR}/../bin/FileCheck")
  set(LIT_ARGS "-Dfilecheck=${FILECHECK_PATH}")
  add_lit_testsuite(check-triton-tle-lit-tests "Running the triton-tle regression tests"
    ${tle_test_binary_dir}
    ARGS ${LIT_ARGS}
    DEPENDS ${TRITON_TLE_TEST_DEPENDS}
  )

  set_target_properties(check-triton-tle-lit-tests PROPERTIES FOLDER "Tests")

  add_lit_testsuites(TRITON-TLE-LIT-TESTS ${tle_test_binary_dir}
    DEPENDS ${TRITON_TLE_TEST_DEPENDS})
endfunction()

cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL tle_add_lit_tests)
