// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @tensor_descriptor_gather_rows
// CHECK-NOT: tt.descriptor_gather
// CHECK: %[[IDX_VIEW:.*]] = memref.reinterpret_cast %{{.*}} to offset: [0], sizes: [32], strides: [1]
// CHECK: %[[IDX_ALLOC:.*]] = memref.alloc() : memref<32xi32>
// CHECK: memref.copy %[[IDX_VIEW]], %[[IDX_ALLOC]]
// CHECK: %[[XOFFSETS:.*]] = bufferization.to_tensor %[[IDX_ALLOC]] restrict writable : memref<32xi32> to tensor<32xi32>
// CHECK: %[[OUT_ALLOC:.*]] = memref.alloc() : memref<32x32xf32>
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK: %[[XOFFSET:.*]] = tensor.extract %[[XOFFSETS]][%{{.*}}] : tensor<32xi32>
// CHECK: %[[DESC_VIEW:.*]] = memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [1, 32], strides: [128, 1]
// CHECK: %[[OUT_SUBVIEW:.*]] = memref.subview %[[OUT_ALLOC]][%{{.*}}, 0] [1, 32] [1, 1]
// CHECK: memref.copy %{{.*}}, %{{.*}}
// CHECK: %[[OUT_TENSOR:.*]] = bufferization.to_tensor %[[OUT_ALLOC]] restrict writable : memref<32x32xf32> to tensor<32x32xf32>
// CHECK: bufferization.materialize_in_destination %[[OUT_TENSOR]] in writable %{{.*}} : (tensor<32x32xf32>, memref<32x32xf32, strided<[32, 1]>>) -> ()

module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  tt.func public @tensor_descriptor_gather_rows(%out_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %in_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %idx_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %y: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<32x1xi32>
    %desc = arith.constant 1 : i64
    %desc_0 = arith.constant 128 : i64
    %c128_i32 = arith.constant 128 : i32
    %idx = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %idx_1 = tt.splat %idx_ptr : !tt.ptr<i32> -> tensor<32x!tt.ptr<i32>>
    %idx_2 = tt.addptr %idx_1, %idx : tensor<32x!tt.ptr<i32>>, tensor<32xi32>
    %idx_3 = tt.load %idx_2 : tensor<32x!tt.ptr<i32>>
    %desc_4 = tt.make_tensor_descriptor %in_ptr, [%c128_i32, %c128_i32], [%desc_0, %desc] : <f32>, <tensor<1x32xf32>>
    %out = tt.descriptor_gather %desc_4[%idx_3, %y] : (!tt.tensordesc<tensor<1x32xf32>>, tensor<32xi32>, i32) -> tensor<32x32xf32>
    %0 = tt.expand_dims %idx {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %1 = arith.muli %0, %cst : tensor<32x1xi32>
    %2 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<32x1x!tt.ptr<f32>>
    %3 = tt.addptr %2, %1 : tensor<32x1x!tt.ptr<f32>>, tensor<32x1xi32>
    %4 = tt.expand_dims %idx {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %5 = tt.broadcast %3 : tensor<32x1x!tt.ptr<f32>> -> tensor<32x32x!tt.ptr<f32>>
    %6 = tt.broadcast %4 : tensor<1x32xi32> -> tensor<32x32xi32>
    %7 = tt.addptr %5, %6 : tensor<32x32x!tt.ptr<f32>>, tensor<32x32xi32>
    tt.store %7, %out : tensor<32x32x!tt.ptr<f32>>
    tt.return
  }
}
