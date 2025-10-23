
llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TRITON_ADAPTER_TEST_DEPENDS
  triton-adapter-opt
)

set(FILECHECK_PATH "${LLVM_LIBRARY_DIR}/../bin/FileCheck")
set(LIT_ARGS "-Dfilecheck=${FILECHECK_PATH}")

add_lit_testsuite(check-triton-adapter-lit-tests "Running the triton-adapter regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${LIT_ARGS}
  DEPENDS ${TRITON_ADAPTER_TEST_DEPENDS}
  )

set_target_properties(check-triton-adapter-lit-tests PROPERTIES FOLDER "Tests")

add_lit_testsuites(TRITON_ADAPTER_TIT_TESTS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TRITON_ADAPTER_TEST_DEPENDS})
