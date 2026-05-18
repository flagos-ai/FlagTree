set(LLVM_TARGET_DEFINITIONS Passes.td)
if(FLAGTREE_MTHREADS_TLE)
  set(_TLE_TABLEGEN_DEFS -D__TLE__)
else()
  set(_TLE_TABLEGEN_DEFS "")
endif()

mlir_tablegen(Passes.h.inc -gen-pass-decls -name TritonMUSAGPU
              ${_TLE_TABLEGEN_DEFS})
add_public_tablegen_target(TritonMUSAGPUTransformsIncGen)
