//===----------------------- Passes.h -------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LEGALIZE_TENSOR_FORM_LOOPS_CONVERSION_PASSES_H
#define LEGALIZE_TENSOR_FORM_LOOPS_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "magic-kernel/Conversion/LegalizeTensorFormLoops/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // LEGALIZE_TENSOR_FORM_LOOPS_CONVERSION_PASSES_H
