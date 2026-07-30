// RUN: not triton-shared-opt --split-input-file --triton-to-linalg-experimental %s 2>&1 | FileCheck %s
module {
  tt.func public @make_tensor_ptr_error(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_program_id y : i32
    %2 = arith.muli %0, %c128_i32 : i32
    %3 = arith.muli %1, %c128_i32 : i32
    %4 = arith.extsi %arg2 : i32 to i64
    %5 = tt.make_tensor_ptr %arg0, [%4, %4], [%c1_i64, %4], [%2, %3] {order = array<i32: 0, 1>} : <tensor<128x128xf32>>
    %6 = tt.make_tensor_ptr %arg1, [%4, %4], [%4, %c1_i64], [%3, %2] {order = array<i32: 1, 0>} : <tensor<128x128xf32>>
    %7 = tt.load %5 : !tt.ptr<tensor<128x128xf32>>
    tt.store %6, %7 : !tt.ptr<tensor<128x128xf32>>
    tt.return
  }
}

// CHECK: error: non-decreasing dimension order on tensor pointers are not yet supported
// CHECK-NEXT: %5 = tt.make_tensor_ptr