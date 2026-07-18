
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TritonGCUConversion)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix TritonGCUConversion)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix TritonGCUConversion)
add_mlir_doc(Passes TritonGCUConversionPasses_${arch} conversions/ -gen-pass-doc)

add_public_tablegen_target(MLIRTritonGCUConversionPassIncGen_${arch})
