//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_LINALG_FUSION_PASSES_H
#define TRITON_CONVERSION_LINALG_FUSION_PASSES_H

#include "tsingmicro-tx81/Conversion/LinalgFusion/LinalgFusion.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "tsingmicro-tx81/Conversion/LinalgFusion/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_LINALG_FUSION_PASSES_H
