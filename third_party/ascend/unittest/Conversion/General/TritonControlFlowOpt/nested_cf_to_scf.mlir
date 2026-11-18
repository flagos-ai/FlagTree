// RUN: triton-opt --triton-control-flow-opt --split-input-file %s | FileCheck %s --implicit-check-not="cf.cond_br" --implicit-check-not="cf.br"

module {
  tt.func public @nested_cf_to_scf(%outer: i1, %inner: i1, %x: i32, %y: i32, %z: i32) -> i32 {
    cf.cond_br %outer, ^bb1, ^bb2
  ^bb1:
    cf.cond_br %inner, ^bb3, ^bb4
  ^bb2:
    %outer_else = arith.subi %y, %z : i32
    cf.br ^bb6(%outer_else : i32)
  ^bb3:
    %inner_then = arith.addi %x, %z : i32
    cf.br ^bb5(%inner_then : i32)
  ^bb4:
    %inner_else = arith.subi %x, %z : i32
    cf.br ^bb5(%inner_else : i32)
  ^bb5(%inner_result: i32):
    %then_result = arith.addi %inner_result, %y : i32
    cf.br ^bb6(%then_result : i32)
  ^bb6(%result: i32):
    tt.return %result : i32
  }
}

// CHECK-LABEL: tt.func public @nested_cf_to_scf
// CHECK:       %[[OUTER_IF:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:         %[[INNER_IF:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:           arith.addi
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         } else {
// CHECK:           arith.subi
// CHECK:           scf.yield %{{.*}} : i32
// CHECK:         }
// CHECK:         arith.addi %[[INNER_IF]]
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       } else {
// CHECK:         arith.subi
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       }
// CHECK:       tt.return %[[OUTER_IF]] : i32
