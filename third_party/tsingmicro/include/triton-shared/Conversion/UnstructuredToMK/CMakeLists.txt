#===------------------------------------------------------------------------===#
#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
#===------------------------------------------------------------------------===#

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name UnstructuredToMK)
add_public_tablegen_target(UnstructuredToMKConversionPassIncGen)
