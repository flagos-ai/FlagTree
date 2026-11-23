// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @test_parse_select_dense_const
// CHECK: %[[C32_I32:.*]] = arith.constant 32 : i32
// CHECK: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK: %[[CMP:.*]] = arith.cmpi sgt, %{{.*}}, %[[C0_I32]] : i32
// CHECK: %[[SELECT:.*]] = arith.select %[[CMP]], %[[C0_I32]], %[[C32_I32]] : i32
// CHECK: %[[IDX_CAST:.*]] = arith.index_cast %[[SELECT]] : i32 to index
// CHECK: %[[ADD:.*]] = arith.addi %[[IDX_CAST]], %{{.*}} : index

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @test_parse_select_dense_const(%src_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %numel: i32, %flag: i32) attributes {noinline = false} {
    %zero_f32 = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %pid = tt.get_program_id x : i32
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %base = arith.muli %pid, %c8_i32 : i32
    %base_splat = tt.splat %base : i32 -> tensor<8xi32>
    %offs_full = arith.addi %offs, %base_splat : tensor<8xi32>
    %numel_splat = tt.splat %numel : i32 -> tensor<8xi32>
    %mask = arith.cmpi slt, %offs_full, %numel_splat : tensor<8xi32>
    %cond = arith.cmpi sgt, %flag, %c0_i32 : i32
    %const_a = arith.constant dense<0> : tensor<8xi32>
    %const_b = arith.constant dense<32> : tensor<8xi32>
    %chosen = arith.select %cond, %const_a, %const_b : tensor<8xi32>
    %write_offs = arith.addi %chosen, %offs_full : tensor<8xi32>
    %src_ptrs = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %src_addptr = tt.addptr %src_ptrs, %offs_full : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %data = tt.load %src_addptr, %mask, %zero_f32 : tensor<8x!tt.ptr<f32>>
    %dst_ptrs = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %dst_addptr = tt.addptr %dst_ptrs, %write_offs : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %dst_addptr, %data, %mask : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
