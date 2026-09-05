//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LEGALIZE_TENSOR_FORM_LOOPS_CONVERSION_PASSES
#define LEGALIZE_TENSOR_FORM_LOOPS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def LegalizeTensorFormLoops : Pass<"legalize-tensor-form-loops"> {
  let summary = "Legalize tensor dependencies in the loop.";
}

#endif
