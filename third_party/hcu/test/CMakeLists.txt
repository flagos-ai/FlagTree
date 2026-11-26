add_subdirectory(lib)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-triton-hcu "Running the HCU regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS triton-opt
)
