// RUN: triton-opt --triton-control-flow-opt --split-input-file %s | FileCheck %s

module {
  tt.func public @scalar_pointer_scf_iter_arg_not_decoupled(%base: !tt.ptr<f32>) -> !tt.ptr<f32> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %base) -> (!tt.ptr<f32>) {
      %next = tt.addptr %ptr, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %next : !tt.ptr<f32>
    }
    tt.return %result : !tt.ptr<f32>
  }
}

// CHECK-LABEL: tt.func public @scalar_pointer_scf_iter_arg_not_decoupled
// CHECK:       scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!tt.ptr<f32>)
// CHECK:       tt.addptr
// CHECK:       tt.return %{{.*}} : !tt.ptr<f32>
