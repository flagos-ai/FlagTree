// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_parse_div
// CHECK: %[[SUB:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: ^bb0(%[[IN:.*]]: i32, %[[IN_5:.*]]: i32, %[[OUT:.*]]: i32):
// CHECK: %[[SUBI:.*]] = arith.subi %[[IN]], %[[IN_5]] : i32
// CHECK: linalg.yield %[[SUBI]] : i32

// CHECK: %[[DIV:.*]] = linalg.generic
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK: ^bb0(%[[IN:.*]]: i32, %[[IN_5:.*]]: i32, %[[OUT:.*]]: i32):
// CHECK: %[[DIVI:.*]] = arith.divsi %[[IN]], %[[IN_5]] : i32
// CHECK: linalg.yield %[[DIVI]] : i32

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @test_parse_div(%idx_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %src_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %numel: i32, %bias: i32, %stride: i32) attributes {noinline = false} {
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %zero_i32 = arith.constant dense<0> : tensor<8xi32>
    %c8_i32 = arith.constant 8 : i32
    %pid = tt.get_program_id x : i32
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %base = arith.muli %pid, %c8_i32 : i32
    %base_splat = tt.splat %base : i32 -> tensor<8xi32>
    %offs_full = arith.addi %offs, %base_splat : tensor<8xi32>
    %numel_splat = tt.splat %numel : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %offs_full, %numel_splat : tensor<8xi32>
    %idx_ptrs = tt.splat %idx_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %idx_addptr = tt.addptr %idx_ptrs, %offs_full : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %indices = tt.load %idx_addptr, %mask, %zero_i32 : tensor<8x!tt.ptr<i32>>
    %bias_splat = tt.splat %bias : i32 -> tensor<8xi32>
    %subbed = arith.subi %indices, %bias_splat : tensor<8xi32>
    %bias_splat2 = tt.splat %bias : i32 -> tensor<8xi32>
    %stride_splat = tt.splat %stride : i32 -> tensor<8xi32>
    %scalar_div = arith.divsi %bias_splat2, %stride_splat : tensor<8xi32>
    %combined = arith.addi %subbed, %scalar_div : tensor<8xi32>
    %src_ptrs = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %src_addptr = tt.addptr %src_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %data = tt.load %src_addptr, %mask, %zero_f32 : tensor<8x!tt.ptr<f32>>
    %dst_ptrs = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %dst_addptr = tt.addptr %dst_ptrs, %combined : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %dst_addptr, %data, %mask : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
