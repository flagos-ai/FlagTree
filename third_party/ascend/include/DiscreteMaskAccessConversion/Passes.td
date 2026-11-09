/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef DISCRETE_MASK_ACCESS_CONVERSION_PASSES
#define DISCRETE_MASK_ACCESS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def DiscreteMaskAccessConversion : Pass<"discrete-mask-access-conversion", "mlir::ModuleOp"> {
  let summary = "Recognize and convert discrete mask memory access";
  let constructor = "triton::createDiscreteMaskAccessConversionPass()";
  let options = [
    Option<"compileOn91095", "compile-on-910-95",
        "bool", /*default*/"false",
        "compile on 910_95">,
    Option<"compileMode", "compile-mode",
        "std::string", /*default*/"\"simd_simt_template\"",
        "Compilation mode: simd, simd_simt_template (or unstructured_in_simt), simt_only">
  ];
}

#endif // DISCRETE_MASK_ACCESS_CONVERSION_PASSES
