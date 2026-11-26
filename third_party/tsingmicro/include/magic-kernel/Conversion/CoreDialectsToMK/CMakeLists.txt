#===------------------------------------------------------------------------===#
#
# Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
# All rights reserved.
#
#===------------------------------------------------------------------------===#

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name CoreDialectsMK)
add_public_tablegen_target(CoreDialectsToMKConversionPassIncGen)
