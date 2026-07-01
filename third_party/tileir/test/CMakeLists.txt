# Passed to lit.site.cfg.py.in to set up the path where to find libraries.
set(MLIR_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TRITON_CUDA_TILE_TEST_DEPENDS
  triton-cuda-tile-opt
  triton-opt
  triton-llvm-opt
)

set(FILECHECK_PATH "${LLVM_LIBRARY_DIR}/../bin/FileCheck")
set(LIT_ARGS "-Dfilecheck=${FILECHECK_PATH}")

# This target can be used to just build the dependencies
# for the check-triton-cuda-tile target without executing the tests.
# This is useful for bots when splitting the build step from
# the testing step.
add_custom_target(check-triton-cuda-tile-build-only
  DEPENDS ${TRITON_CUDA_TILE_TEST_DEPENDS}
)

add_lit_testsuite(check-triton-cuda-tile "Running the triton-cuda-tile tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TRITON_CUDA_TILE_TEST_DEPENDS}
  )
set_target_properties(check-triton-cuda-tile PROPERTIES FOLDER "Tests")

add_lit_testsuites(triton-cuda-tile ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${TRITON_CUDA_TILE_TEST_DEPENDS}
) 
