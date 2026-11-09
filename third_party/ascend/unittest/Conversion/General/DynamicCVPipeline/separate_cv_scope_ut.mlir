// RUN: triton-opt %s --separate-cv-scope | FileCheck %s

// Regression for the structural-user erase crash in SeparateCVScope:
// a VECTOR-only scf.if feeds the iter_args of a mixed VECTOR/CUBE scf.for.
// In the CUBE clone the scf.if shell becomes semantically dead before the loop
// is fully rewritten, so the pass must not erase the shell while the raw SSA
// use from the downstream scf.for still exists.

// CHECK-LABEL: func.func @if_results_feed_followup_loop_structural_user_repro(
// CHECK: scope.scope : () -> () {
// CHECK: scf.if
// CHECK: scf.for
// CHECK: tensor.extract
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
// CHECK: scope.scope : () -> () {
// CHECK: scf.for
// CHECK: memref.store
// CHECK: scope.return
// CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
module {
  func.func @if_results_feed_followup_loop_structural_user_repro(
      %lb: i32,
      %ub: i32,
      %cond: i1,
      %vec0: tensor<4xf32>,
      %vec1: tensor<4xf32>,
      %cube_init: i32,
      %outv: memref<1xf32>,
      %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c1v = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %c1c = arith.constant {ssbuffer.core_type = "CUBE"} 1 : i32

    %0:2 = scf.if %cond -> (tensor<4xf32>, tensor<4xf32>) {
      %1 = arith.addf %vec0, %vec1 {ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
      %2 = arith.mulf %1, %vec1 {ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
      scf.yield {ssbuffer.core_type = "VECTOR, VECTOR"} %1, %2 : tensor<4xf32>, tensor<4xf32>
    } else {
      scf.yield {ssbuffer.core_type = "VECTOR, VECTOR"} %vec0, %vec1 : tensor<4xf32>, tensor<4xf32>
    } {ssbuffer.core_type = "VECTOR, VECTOR"}

    %1:3 = scf.for %i = %lb to %ub step %c1v iter_args(%lhs = %0#0, %rhs = %0#1, %cube = %cube_init) -> (tensor<4xf32>, tensor<4xf32>, i32) : i32 {
      %2 = arith.addi %cube, %c1c {ssbuffer.core_type = "CUBE"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR, VECTOR, CUBE"} %lhs, %rhs, %2 : tensor<4xf32>, tensor<4xf32>, i32
    } {ssbuffer.core_type = "VECTOR, VECTOR, CUBE"}

    %2 = tensor.extract %1#0[%idxv] {ssbuffer.core_type = "VECTOR"} : tensor<4xf32>
    memref.store %2, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xf32>
    memref.store %1#2, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
    func.return
  }

  // A while induction variable can belong to VECTOR while still controlling
  // the trip count of CUBE work. Both separated scopes must preserve the
  // predicate-carried value and its update.

  // CHECK-LABEL: func.func @while_predicate_controls_mixed_scope(
  // CHECK-SAME: %[[UB:.*]]: i32
  // CHECK: scope.scope : () -> () {
  // CHECK: scf.while (%[[V_IV:.*]] = %{{.*}}) : (i32) -> i32 {
  // CHECK: %[[V_COND:.*]] = arith.cmpi slt, %[[V_IV]], %[[UB]]
  // CHECK: scf.condition(%[[V_COND]]) %[[V_IV]] : i32
  // CHECK: ^bb0(%[[V_BODY_IV:.*]]: i32):
  // CHECK: %[[V_NEXT:.*]] = arith.addi %[[V_BODY_IV]], %{{.*}}
  // CHECK: scf.yield %[[V_NEXT]] : i32
  // CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<VECTOR>}
  // CHECK: scope.scope : () -> () {
  // CHECK: scf.while (%[[C_IV:.*]] = %{{.*}}) : (i32) -> i32 {
  // CHECK: %[[C_COND:.*]] = arith.cmpi slt, %[[C_IV]], %[[UB]]
  // CHECK: scf.condition(%[[C_COND]]) %[[C_IV]] : i32
  // CHECK: ^bb0(%[[C_BODY_IV:.*]]: i32):
  // CHECK: memref.store
  // CHECK: %[[C_NEXT:.*]] = arith.addi %[[C_BODY_IV]], %{{.*}}
  // CHECK: scf.yield %[[C_NEXT]] : i32
  // CHECK: } {hivm.matmul_limited_in_cube, hivm.tcore_type = #hivm.tcore_type<CUBE>}
  func.func @while_predicate_controls_mixed_scope(
      %ub: i32, %outv: memref<1xi32>, %outc: memref<1xi32>) {
    %idxv = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : index
    %idxc = arith.constant {ssbuffer.core_type = "CUBE"} 0 : index
    %c0 = arith.constant {ssbuffer.core_type = "VECTOR"} 0 : i32
    %c1 = arith.constant {ssbuffer.core_type = "VECTOR"} 1 : i32
    %cube_value = arith.constant {ssbuffer.core_type = "CUBE"} 7 : i32

    %result = scf.while (%iv = %c0) : (i32) -> i32 {
      %cond = arith.cmpi slt, %iv, %ub {ssbuffer.core_type = "VECTOR"} : i32
      scf.condition(%cond) {ssbuffer.core_type = "VECTOR"} %iv : i32
    } do {
    ^bb0(%iv: i32):
      memref.store %cube_value, %outc[%idxc] {ssbuffer.core_type = "CUBE"} : memref<1xi32>
      %next = arith.addi %iv, %c1 {ssbuffer.core_type = "VECTOR"} : i32
      scf.yield {ssbuffer.core_type = "VECTOR"} %next : i32
    } attributes {ssbuffer.core_type = "VECTOR"}

    memref.store %result, %outv[%idxv] {ssbuffer.core_type = "VECTOR"} : memref<1xi32>
    func.return
  }
}
