// RUN: triton-opt '--add_dynamic_cv_pipeline=compile-on-910-95=True' --debug-only=add-dynamic-cv-pipeline %s 2>&1 | FileCheck %s --implicit-check-not='note: see current operation:'

// Test: Module with no linalg.matmul. PreCheckMatmul sets ERRCODE_IGNORED,
// AddDynamicCVPipeline falls back with a "Kernel not applicable..." warning
// that does NOT include the full module IR dump.

// CHECK: warning: [add-dynamic-cv-pipeline] Kernel not applicable for dynamic CV pipeline
module {
  func.func @fallback_no_matmul(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    return %0 : i32
  }
}
