// RUN: triton-opt '--add_dynamic_cv_pipeline=compile-on-910-95=True' %s | FileCheck %s

// CHECK-LABEL: module attributes {triton_ascend.dynamic_cv_pipeline.rc = 2 : i32}
module {
  // CHECK-LABEL: func.func @fallback_existing_scope
  // CHECK: arith.addi
  // CHECK: scope.scope
  // CHECK: scope.return
  // CHECK: return
  func.func @fallback_existing_scope(%arg0: i32, %arg1: i32) -> i32 {
    %0 = arith.addi %arg0, %arg1 : i32
    scope.scope : () -> () {
      scope.return
    }
    return %0 : i32
  }
}
