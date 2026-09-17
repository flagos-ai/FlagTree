//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef TLE_TO_MK_CONVERSION_PASSES_H
#define TLE_TO_MK_CONVERSION_PASSES_H

#include "magic-kernel/Conversion/TLEToMK/TLEToMK.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "magic-kernel/Conversion/TLEToMK/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TLE_TO_MK_CONVERSION_PASSES_H
