set(LLVM_TARGET_DEFINITIONS Passes.td)
if(FLAGTREE_MTHREADS_TLE)
  set(_TLE_TABLEGEN_DEFS -D__TLE__)
  set(_TLE_TABLEGEN_DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/../../../tle/dialect/include/MUSATLE/Transforms/Passes.td)
else()
  set(_TLE_TABLEGEN_DEFS "")
  set(_TLE_TABLEGEN_DEPENDS "")
endif()

set(LLVM_TARGET_DEPENDS ${_TLE_TABLEGEN_DEPENDS})
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TritonMUSAGPU
              ${_TLE_TABLEGEN_DEFS})
add_public_tablegen_target(TritonMUSAGPUTransformsIncGen)
