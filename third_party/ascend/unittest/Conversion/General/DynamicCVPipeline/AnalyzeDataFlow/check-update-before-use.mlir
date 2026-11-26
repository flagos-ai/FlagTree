// RUN: triton-opt --split-input-file --analyze-args %s | FileCheck %s

// Buffer count is set to 2/1 on every module so the existing
// `checkTensorArgsInMainLoop` gate (intra=3 && inter=2) is disabled; these
// tests exercise only the new update-before-use check.

// Test 1 (positive, scf.for): tensor iter_arg %arg34 is used twice.
// The defining op %82 of yield's i=0 operand is the iter_arg's update; it
// is defined BEFORE %83 (update-before-use) and at a different block_id
// (10 vs 8). This must trigger ERRCODE_IGNORED.
// CHECK: triton_ascend.dynamic_cv_pipeline.rc = 2
// CHECK: @update_before_use_for
module attributes {ssbuffer.inter_core_buf_count = 1 : i32, ssbuffer.intra_buf_count = 2 : i32} {
  func.func @update_before_use_for() -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %init = tensor.empty() : tensor<64x64xf32>
    %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg34 = %init) -> (tensor<64x64xf32>) {
      // First use of %arg34 at block_id=10 (before update).
      %81 = arith.mulf %arg34, %init {ssbuffer.block_id = 10 : i32} : tensor<64x64xf32>
      // Update-defining op at block_id=10.
      %82 = arith.addf %81, %init {ssbuffer.block_id = 10 : i32} : tensor<64x64xf32>
      // Late use of %arg34 AFTER update at block_id=8 — different block_id.
      %83 = arith.mulf %arg34, %init {ssbuffer.block_id = 8 : i32} : tensor<64x64xf32>
      scf.yield %82 : tensor<64x64xf32>
    } {ssbuffer.main_loop = 0 : i32}
    return
  }
}

// -----

// Test 2 (positive, scf.while): same update-before-use pattern as Test 1
// but on scf.while's do-region. ERRCODE_IGNORED must fire.
// CHECK: triton_ascend.dynamic_cv_pipeline.rc = 2
// CHECK: @update_before_use_while
module attributes {ssbuffer.inter_core_buf_count = 1 : i32, ssbuffer.intra_buf_count = 2 : i32} {
  func.func @update_before_use_while() -> () {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %init = tensor.empty() : tensor<64xf32>
    %result:2 = scf.while (%arg2 = %c0_i64, %arg3 = %init) : (i64, tensor<64xf32>) -> (i64, tensor<64xf32>) {
      %cmp = arith.cmpi slt, %arg2, %c1_i64 : i64
      scf.condition(%cmp) %arg2, %arg3 : i64, tensor<64xf32>
    } do {
    ^bb0(%arg2: i64, %arg3: tensor<64xf32>):
      // First use of %arg3 at block_id=10 (before update).
      %mul = arith.mulf %arg3, %arg3 {ssbuffer.block_id = 10 : i32} : tensor<64xf32>
      // Update-defining op at block_id=10.
      %update = arith.addf %mul, %init {ssbuffer.block_id = 10 : i32} : tensor<64xf32>
      // Late use of %arg3 AFTER update at block_id=8 — different block_id.
      %late = arith.mulf %arg3, %init {ssbuffer.block_id = 8 : i32} : tensor<64xf32>
      %next = arith.addi %arg2, %c1_i64 : i64
      scf.yield %next, %update : i64, tensor<64xf32>
    } attributes {ssbuffer.main_loop = 0 : i32}
    return
  }
}

// -----

// Test 3 (negative): tensor iter_arg with code-order update-before-use
// where the late use shares the same block_id as the update. The block_id
// mismatch is what triggers the fallback; with same block_id, no fallback
// must be set on this module.
// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK: @no_update_before_use_same_block
// CHECK: return
module attributes {ssbuffer.inter_core_buf_count = 1 : i32, ssbuffer.intra_buf_count = 2 : i32} {
  func.func @no_update_before_use_same_block() -> () {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %init = tensor.empty() : tensor<64x64xf32>
    %result = scf.for %iv = %c0 to %c10 step %c1 iter_args(%arg34 = %init) -> (tensor<64x64xf32>) {
      %81 = arith.mulf %arg34, %init {ssbuffer.block_id = 10 : i32} : tensor<64x64xf32>
      %82 = arith.addf %81, %init {ssbuffer.block_id = 10 : i32} : tensor<64x64xf32>
      // Late use AFTER update with SAME block_id — must not trigger.
      %83 = arith.mulf %arg34, %init {ssbuffer.block_id = 10 : i32} : tensor<64x64xf32>
      scf.yield %82 : tensor<64x64xf32>
    } {ssbuffer.main_loop = 0 : i32}
    return
  }
}
