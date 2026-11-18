// RUN: triton-opt --triton-to-linalg="named-ops=True" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @index_select_simd_bool
// CHECK: memref.reinterpret_cast {{.*}} : memref<?xi8> to memref<?x8xi8
// CHECK: memref.copy {{.*}} {was_bool_to_int8 = true} : memref<1x8xi8
// CHECK: bufferization.to_tensor {{.*}} {index_select_simd, was_bool_to_int8 = true} : memref<2x8xi8> to tensor<2x8xi8>
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  tt.func public @index_select_simd_bool(%src: !tt.ptr<i8>, %dst: !tt.ptr<i8>) attributes {noinline = false} {
    %c0 = arith.constant 0 : index
    %c-1 = arith.constant -1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %indices = arith.constant dense<[1, 3]> : tensor<2xi32>
    %result = ascend.index_select_simd %src, %indices, 0, [%c4, %c8], [%c-1, %c0], [-1, 8] {was_bool_to_int8 = true} : !tt.ptr<i8>, tensor<2xi32> -> tensor<2x8xi8>
    %offsets = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
    %dsts = tt.splat %dst : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>>
    %dst_ptrs = tt.addptr %dsts, %offsets : tensor<16x!tt.ptr<i8>>, tensor<16xi32>
    %flat = tt.reshape %result : tensor<2x8xi8> -> tensor<16xi8>
    tt.store %dst_ptrs, %flat : tensor<16x!tt.ptr<i8>>
    tt.return
  }
}
