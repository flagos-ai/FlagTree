#===------------------------------------------------------------------------===#
#
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
#
#===------------------------------------------------------------------------===#

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name UnstructuredToMemref)
add_public_tablegen_target(UnstructuredToMemrefConversionPassIncGen)
