// RUN: triton-opt --split-input-file --split-if-by-block-id %s | FileCheck %s

// Verify that kSplittedIf ("ssbuffer.splitted_if") is a unique int tag
// per source if-op: all split if-ops from the same source share the same
// tag, preserved after rearrangeIfOp for downstream passes.

// VCV: then side has VECTOR -> CUBE -> VECTOR with cross-core data flow.
// All 3 split if-ops from this source share tag 0.
// CHECK-LABEL: func.func @has_splitted_if_tag
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK-NOT: ssbuffer.splitted_if = 0 : i32
func.func @has_splitted_if_tag(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    scf.if %cond {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m = linalg.matmul ins(%v1, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v2 = arith.mulf %m, %b {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    }
  }
  return
}

// -----

// CVC: else side has CUBE -> VECTOR -> CUBE. Single source if in this
// test case, counter resets each run, so tag is 0.
// CHECK-LABEL: func.func @has_splitted_if_tag_else
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK-NOT: ssbuffer.splitted_if = 0 : i32
func.func @has_splitted_if_tag_else(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: f32, %d: f32, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : f32
    scf.if %cond {
      %x = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : f32
    } else {
      %m1 = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v1 = arith.addf %m1, %m1 {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m2 = linalg.matmul ins(%a, %v1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 97 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
    }
  }
  return
}

// -----

// Two different source ifs in the same function: first gets tag 0,
// second gets tag 1. Each is VCV then-side (3 split ifs).
// CHECK-LABEL: func.func @two_ifs_different_tags
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK: } {{.*}}ssbuffer.splitted_if = 0 : i32}
// CHECK: } {{.*}}ssbuffer.splitted_if = 1 : i32}
// CHECK: } {{.*}}ssbuffer.splitted_if = 1 : i32}
// CHECK: } {{.*}}ssbuffer.splitted_if = 1 : i32}
// CHECK-NOT: ssbuffer.splitted_if
func.func @two_ifs_different_tags(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    // First if: VCV then-side -> tag 0
    scf.if %cond {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m1 = linalg.matmul ins(%v1, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v2 = arith.mulf %m1, %b {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    }
    // Second if: VCV then-side -> tag 1
    scf.if %cond {
      %v3 = arith.subf %c, %d {ssbuffer.block_id = 97 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %m2 = linalg.matmul ins(%v3, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 98 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v4 = arith.mulf %m2, %b {ssbuffer.block_id = 99 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    }
  }
  return
}
