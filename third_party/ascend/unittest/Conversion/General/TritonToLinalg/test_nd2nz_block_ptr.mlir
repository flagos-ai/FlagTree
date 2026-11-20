// RUN: triton-opt --triton-to-linalg=enable-nd2nz-on-vector=true %s | FileCheck %s

// CHECK-LABEL: func.func @test_nd2nz_block_ptr

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @test_nd2nz_block_ptr(
      %base_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c32_i64 = arith.constant 32 : i64
    %load_ptr = tt.make_tensor_ptr %base_ptr, [%c128_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<4x32xf32>>
    %data = tt.load %load_ptr {boundaryCheck = array<i32>} : !tt.ptr<tensor<4x32xf32>>
    %store_ptr = tt.make_tensor_ptr %out_ptr, [%c128_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<4x32xf32>>
    tt.store %store_ptr, %data {boundaryCheck = array<i32>} : !tt.ptr<tensor<4x32xf32>>
    tt.return
  }
}
