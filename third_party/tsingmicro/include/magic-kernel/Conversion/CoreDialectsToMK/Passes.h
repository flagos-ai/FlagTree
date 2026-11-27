//===------------------- CoreDialectsToMK.h -------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Wrap all the conversion from core dialects to backend dialects(MK etc).
//
//===----------------------------------------------------------------------===//

#ifndef CORE_DIALECTS_TO_MK_CONVERSION_PASSES_H
#define CORE_DIALECTS_TO_MK_CONVERSION_PASSES_H

#include "magic-kernel/Conversion/CoreDialectsToMK/CoreDialectsToMK.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "magic-kernel/Conversion/CoreDialectsToMK/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // CORE_DIALECTS_TO_MK_CONVERSION_PASSES_H
