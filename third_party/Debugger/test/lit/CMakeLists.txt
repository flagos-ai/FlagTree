llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(FLAGTREE_DEBUGGER_LIT_DEPENDS triton-opt)
set(FILECHECK_PATH "${LLVM_LIBRARY_DIR}/../bin/FileCheck")
set(LIT_ARGS "-Dfilecheck=${FILECHECK_PATH}")

add_lit_testsuite(
  check-flagtree-debugger-lit-tests
  "Running the FlagTree debugger regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${LIT_ARGS}
  DEPENDS ${FLAGTREE_DEBUGGER_LIT_DEPENDS}
)

set_target_properties(
  check-flagtree-debugger-lit-tests
  PROPERTIES FOLDER "Tests"
)

add_lit_testsuites(
  FLAGTREE-DEBUGGER-LIT-TESTS
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${FLAGTREE_DEBUGGER_LIT_DEPENDS}
)
