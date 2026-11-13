configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)
set(TRITON_PPU_TEST_DEPENDS
  triton-opt
)

# This target can be used to just build the dependencies for the
# check-triton-ppu target without executing the tests.
add_custom_target(check-triton-ppu-build-only
  DEPENDS ${TRITON_PPU_TEST_DEPENDS}
)

add_lit_testsuite(check-triton-ppu "Running the triton-ppu regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TRITON_PPU_TEST_DEPENDS}
  )
set_target_properties(check-triton-ppu PROPERTIES FOLDER "Tests")
add_lit_testsuites(TRITON-PPU-LIT-TESTS ${CMAKE_CURRENT_BINARY_DIR} DEPENDS ${TRITON_PPU_TEST_DEPENDS})
