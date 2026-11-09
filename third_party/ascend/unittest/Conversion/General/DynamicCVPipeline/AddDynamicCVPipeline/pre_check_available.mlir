// RUN: triton-opt %s --pre-check-dynamic-cv-pipeline-available | FileCheck %s --check-prefix=PASS

// PASS-LABEL: func.func @pre_check_available_passes(
// PASS: linalg.matmul
// PASS-NOT: triton_ascend.dynamic_cv_pipeline.rc
module {
  func.func @pre_check_available_passes(
      %arg0: memref<16x16xf16>,
      %arg1: memref<16x16xf16>,
      %arg2: memref<16x16xf32>) {
    linalg.matmul
      ins(%arg0, %arg1 : memref<16x16xf16>, memref<16x16xf16>)
      outs(%arg2 : memref<16x16xf32>)
    return
  }
}
