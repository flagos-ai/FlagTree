// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @sub_offset_kernel

// CHECK: memref.subview {{.*}} : memref<8xi32, strided<[1], offset: ?>> to memref<?xi32, strided<[1], offset: ?>>
// CHECK: memref.subview {{.*}} : memref<8xi32> to memref<?xi32, strided<[1]>>
// CHECK: memref.copy

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @sub_offset_kernel(%idx_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %src_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %dst_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %numel: i32, %correction: i32) attributes {noinline = false} {
    %val = arith.constant dense<0.000000e+00> : tensor<8xf32>
    %base_idx = arith.constant dense<0> : tensor<8xi32>
    %c8_i32 = arith.constant 8 : i32
    %pid = tt.get_program_id x : i32
    %offs = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %offs_0 = arith.muli %pid, %c8_i32 : i32
    %offs_1 = tt.splat %offs_0 : i32 -> tensor<8xi32>
    %offs_2 = arith.addi %offs, %offs_1 : tensor<8xi32>
    %mask = tt.splat %numel : i32 -> tensor<8xi32>
    %mask_3 = arith.cmpi slt, %offs_2, %mask : tensor<8xi32>
    %base_idx_4 = tt.splat %idx_ptr : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %base_idx_5 = tt.addptr %base_idx_4, %offs_2 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %base_idx_6 = tt.load %base_idx_5, %mask_3, %base_idx : tensor<8x!tt.ptr<i32>>
    %adjusted = tt.splat %correction : i32 -> tensor<8xi32>
    %adjusted_7 = arith.subi %base_idx_6, %adjusted : tensor<8xi32>
    %val_8 = tt.splat %src_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %val_9 = tt.addptr %val_8, %offs_2 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    %val_10 = tt.load %val_9, %mask_3, %val : tensor<8x!tt.ptr<f32>>
    %0 = tt.splat %dst_ptr : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>>
    %1 = tt.addptr %0, %adjusted_7 : tensor<8x!tt.ptr<f32>>, tensor<8xi32>
    tt.store %1, %val_10, %mask_3 : tensor<8x!tt.ptr<f32>>
    tt.return
  }
}
