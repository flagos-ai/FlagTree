// RUN: triton-opt --triton-control-flow-opt --split-input-file %s | FileCheck %s --implicit-check-not="cf.cond_br" --implicit-check-not="cf.br"

module {
  tt.func public @terminal_return_void(%cond: i1) {
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    tt.return
  ^bb2:
    cf.br ^bb3
  ^bb3:
    tt.return
  }
}

// CHECK-LABEL: tt.func public @terminal_return_void
// CHECK:       scf.if %{{.*}} {
// CHECK:       } else {
// CHECK:       }
// CHECK:       tt.return

// -----

module {
  tt.func public @terminal_return_value(%cond: i1, %lhs: i32, %rhs: i32) -> i32 {
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %then = arith.addi %lhs, %rhs : i32
    tt.return %then : i32
  ^bb2:
    %else = arith.subi %lhs, %rhs : i32
    tt.return %else : i32
  }
}

// CHECK-LABEL: tt.func public @terminal_return_value
// CHECK:       %[[IF:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:         arith.addi
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       } else {
// CHECK:         arith.subi
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       }
// CHECK:       tt.return %[[IF]] : i32

// -----

module {
  tt.func public @terminal_return_nested(%outer: i1, %inner: i1, %x: i32, %y: i32, %z: i32) -> i32 {
    cf.cond_br %outer, ^bb1, ^bb2
  ^bb1:
    %then = arith.addi %x, %y : i32
    tt.return %then : i32
  ^bb2:
    cf.cond_br %inner, ^bb3, ^bb4
  ^bb3:
    %inner_then = arith.addi %x, %z : i32
    tt.return %inner_then : i32
  ^bb4:
    %inner_else = arith.subi %x, %z : i32
    tt.return %inner_else : i32
  }
}

// CHECK-LABEL: tt.func public @terminal_return_nested
// CHECK:       %[[OUTER:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:         arith.addi
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       } else {
// CHECK:         %[[INNER:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:           arith.addi
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         } else {
// CHECK:           arith.subi
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         }
// CHECK:         scf.yield %[[INNER]] : i32
// CHECK:       }
// CHECK:       tt.return %[[OUTER]] : i32

// -----

module {
  tt.func private @entry_return_with_unreachable_block() -> tensor<32xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf32>
    tt.return %cst : tensor<32xf32>
  ^bb1:
    %poison = ub.poison : tensor<32xf32>
    tt.return %poison : tensor<32xf32>
  }
}

// CHECK-LABEL: tt.func private @entry_return_with_unreachable_block
// CHECK:       tt.return
// CHECK-NOT:   ^bb1

// -----

module {
  tt.func public @terminal_return_branch_args(%cond: i1, %lhs: i32, %rhs: i32, %bias: i32) -> i32 {
    cf.cond_br %cond, ^bb1(%lhs : i32), ^bb2(%rhs : i32)
  ^bb1(%v1: i32):
    %then = arith.addi %v1, %bias : i32
    tt.return %then : i32
  ^bb2(%v2: i32):
    %else = arith.subi %v2, %bias : i32
    tt.return %else : i32
  }
}

// CHECK-LABEL: tt.func public @terminal_return_branch_args
// CHECK:       %[[IF:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:         arith.addi
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       } else {
// CHECK:         arith.subi
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       }
// CHECK:       tt.return %[[IF]] : i32

// -----

module {
  tt.func public @terminal_return_nested_with_join(%outer: i1, %inner: i1, %x: i32, %y: i32, %z: i32) -> i32 {
    cf.cond_br %outer, ^bb1(%x : i32), ^bb_join(%y : i32)
  ^bb1(%v: i32):
    cf.cond_br %inner, ^bb_ret, ^bb_join(%v : i32)
  ^bb_ret:
    %early = arith.addi %x, %z : i32
    tt.return %early : i32
  ^bb_join(%j: i32):
    %cont = arith.subi %j, %z : i32
    tt.return %cont : i32
  }
}

// CHECK-LABEL: tt.func public @terminal_return_nested_with_join
// CHECK:       %[[OUTER:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:         %[[INNER:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:           arith.addi
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         } else {
// CHECK:           arith.subi
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         }
// CHECK:         scf.yield %[[INNER]] : i32
// CHECK:       } else {
// CHECK:         arith.subi
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       }
// CHECK:       tt.return %[[OUTER]] : i32
