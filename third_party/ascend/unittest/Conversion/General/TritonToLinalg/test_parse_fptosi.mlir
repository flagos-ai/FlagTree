// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_parse_fptosi
// CHECK: %[[GENERIC:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: ^bb0(%[[IN:.*]]: f32, %[[OUT:.*]]: i32):
// CHECK: %[[FPTOSI:.*]] = arith.fptosi %[[IN]] : f32 to i32
// CHECK: linalg.yield %[[FPTOSI]] : i32

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @test_parse_fptosi(%fidx_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %src_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %numel: i32) attributes {noinline = false} {
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %c8_i32 = arith.constant 8 : i32
    %pid = tt.get_program_id x : i32
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %base = arith.muli %pid, %c8_i32 : i32
    %base_splat = tt.splat %base : i32 -> tensor<8xi32>
    %offs_full = arith.addi %offs, %base_splat : tensor<8xi32>
    %numel_splat = tt.splat %numel : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %offs_full, %numel_splat : tensor<8xi32>
    %offs_i64 = arith.extsi %offs_full : tensor<8xi32> to tensor<8xi64>
    %fidx_ptrs = tt.splat %fidx_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %fidx_addptr = tt.addptr %fidx_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %float_indices = tt.load %fidx_addptr, %mask, %zero_f32 : tensor<8x!tt.ptr<f32>>
    %int_indices = arith.fptosi %float_indices : tensor<8xf32> to tensor<8xi32>
    %src_ptrs = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %src_addptr = tt.addptr %src_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %data = tt.load %src_addptr, %mask, %zero_f32 : tensor<8x!tt.ptr<f32>>
    %dst_ptrs = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %dst_addptr = tt.addptr %dst_ptrs, %int_indices : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %dst_addptr, %data, %mask : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
