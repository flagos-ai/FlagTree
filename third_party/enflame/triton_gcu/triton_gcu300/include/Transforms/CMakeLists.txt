
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name TritonGCUTransforms)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix TritonGCUTransforms)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix TritonGCUTransforms)
add_mlir_doc(Passes TritonGCUTransformsPasses_${arch} transforms/ -gen-pass-doc)

add_public_tablegen_target(TritonGCUTransformsPassIncGen_${arch})
