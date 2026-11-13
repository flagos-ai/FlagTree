// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @tensor_descriptor_scatter_rows_kernel
// CHECK-NOT: tt.descriptor_scatter
// CHECK: %[[C128:.*]] = arith.constant 128 : index
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: %[[XOFFSET:.*]] = tensor.extract %{{.*}}[%{{.*}}] : tensor<32xi32>
// CHECK: %[[BASE:.*]] = arith.muli %{{.*}}, %[[C128]] : index
// CHECK: %[[VIEW:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [1, 32], strides: [128, 1]
// CHECK: %[[ROW:.*]] = tensor.extract_slice %{{.*}}[%{{.*}}, 0] [1, 32] [1, 1] : tensor<32x32xf32> to tensor<1x32xf32>
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[VIEW]]
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[SUBVIEW]]

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @tensor_descriptor_scatter_rows_kernel(%out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %idx_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %y: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %desc = arith.constant 1 : i64
    %desc_0 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %data = arith.constant dense<32> : tensor<32x1xi32>
    %idx = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %idx_1 = tt.splat %idx_ptr : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %idx_2 = tt.addptr %idx_1, %idx : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    %idx_3 = tt.load %idx_2 : tensor<32x!tt.ptr<i32>>
    %data_4 = tt.expand_dims %idx {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %data_5 = arith.muli %data_4, %data : tensor<32x1xi32>
    %data_6 = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %data_7 = tt.addptr %data_6, %data_5 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %data_8 = tt.expand_dims %idx {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %data_9 = tt.broadcast %data_7 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x32x!tt.ptr<f32>>
    %data_10 = tt.broadcast %data_8 : tensor<1x32xi32> -> tensor<32x32xi32>
    %data_11 = tt.addptr %data_9, %data_10 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    %data_12 = tt.load %data_11 : tensor<32x32x!tt.ptr<f32>>
    %desc_13 = tt.make_tensor_descriptor %out_ptr, [%c128_i32, %c128_i32], [%desc_0, %desc] : <f32>, <tensor<1x32xf32>>
    tt.descriptor_scatter %desc_13[%idx_3, %y], %data_12 : !tt.tensordesc<tensor<1x32xf32>>, tensor<32xi32>, i32, tensor<32x32xf32>
    tt.return
  }
}
