// RUN: triton-opt --triton-control-flow-opt --split-input-file %s | FileCheck %s --implicit-check-not="cf.cond_br" --implicit-check-not="cf.br"

module {
  tt.func public @cf_to_scf_result_types(%cond: i1, %lhs: i32, %rhs: i32) -> i32 {
    cf.cond_br %cond, ^bb1, ^bb2
  ^bb1:
    %then = arith.addi %lhs, %rhs : i32
    cf.br ^bb3(%then : i32)
  ^bb2:
    %else = arith.subi %lhs, %rhs : i32
    cf.br ^bb3(%else : i32)
  ^bb3(%result: i32):
    tt.return %result : i32
  }
}

// CHECK-LABEL: tt.func public @cf_to_scf_result_types
// CHECK:       %[[IF:.*]] = scf.if %{{.*}} -> (i32) {
// CHECK:         arith.addi
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       } else {
// CHECK:         arith.subi
// CHECK:         scf.yield %{{.*}} : i32
// CHECK:       }
// CHECK:       tt.return %[[IF]] : i32
