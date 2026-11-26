//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_TO_MK_CONVERSION_PASSES_H
#define LINALG_TO_MK_CONVERSION_PASSES_H

#include "magic-kernel/Conversion/LinalgToMK/LinalgToMK.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "magic-kernel/Conversion/LinalgToMK/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif //  LINALG_TO_MK_CONVERSION_PASSES_H
