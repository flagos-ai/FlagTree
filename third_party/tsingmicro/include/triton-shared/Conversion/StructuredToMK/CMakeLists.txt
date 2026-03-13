message("lyf build in third_party/tsingmicro/include/triton-shared/Conversion/StructuredMK/CMakeLists.txt")
set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name StructuredToMK)
add_public_tablegen_target(StructuredToMKConversionPassIncGen)
