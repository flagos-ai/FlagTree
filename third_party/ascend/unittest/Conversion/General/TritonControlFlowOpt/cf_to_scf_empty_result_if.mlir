// RUN: triton-opt --triton-control-flow-opt --split-input-file %s | FileCheck %s --implicit-check-not="cf.cond_br" --implicit-check-not="cf.br"

module {
  tt.func public @cf_to_scf_empty_result_if(%cond: i1, %lhs: i32, %rhs: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    cf.cond_br %cond, ^bb_return, ^bb_compute
  ^bb_return:
    tt.return %c0_i32 : i32
  ^bb_compute:
    %unused = arith.addi %lhs, %rhs : i32
    cf.br ^bb_return
  }
}

// CHECK-LABEL: tt.func public @cf_to_scf_empty_result_if
// CHECK:       scf.if %{{.*}} {
// CHECK:       } else {
// CHECK:         arith.addi
// CHECK:       }
// CHECK:       tt.return
