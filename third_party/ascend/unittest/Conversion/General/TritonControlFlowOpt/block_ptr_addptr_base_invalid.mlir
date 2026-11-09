// RUN: triton-opt --triton-control-flow-opt %s | FileCheck %s

module {
  tt.func public @direct_addptr_base(%base: !tt.ptr<f32>) {
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %shifted = tt.addptr %base, %c3_i32 : !tt.ptr<f32>, i32
    %ptr = tt.make_tensor_ptr %shifted, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    tt.return
  }

  tt.func public @selected_addptr_base(%base0: !tt.ptr<f32>, %base1: !tt.ptr<f32>, %cond: i1) {
    %c0_i32 = arith.constant 0 : i32
    %c7_i32 = arith.constant 7 : i32
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %shifted = tt.addptr %base0, %c7_i32 : !tt.ptr<f32>, i32
    %selected = arith.select %cond, %shifted, %base1 : !tt.ptr<f32>
    %ptr = tt.make_tensor_ptr %selected, [%c16_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<16xf32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @direct_addptr_base
// CHECK:       %[[SHIFTED:.*]] = tt.addptr
// CHECK:       tt.make_tensor_ptr %[[SHIFTED]],

// CHECK-LABEL: tt.func public @selected_addptr_base
// CHECK:       %[[SHIFTED:.*]] = tt.addptr
// CHECK:       %[[SELECTED:.*]] = arith.select %{{.*}}, %[[SHIFTED]], %{{.*}} : !tt.ptr<f32>
// CHECK:       tt.make_tensor_ptr %[[SELECTED]],
