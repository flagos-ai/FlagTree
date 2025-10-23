// RUN: triton-adapter-opt -split-input-file --triton-to-linalg=named-ops=true %s | FileCheck %s
module {
  tt.func public @fn_broadcast_first_axis(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %3 = tt.load %2 : tensor<32x!tt.ptr<f32>>
    %4 = tt.reshape %3 : tensor<32xf32> -> tensor<1x4x8xf32>
    %5 = tt.broadcast %4 : tensor<1x4x8xf32> -> tensor<128x4x8xf32>
    %6 = tt.reshape %5 : tensor<128x4x8xf32> -> tensor<4096xf32>
    %7 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    tt.store %9, %6 : tensor<4096x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_broadcast_first_axis
// CHECK:           %[[VAL_8:.*]] = arith.constant dense<4096> : tensor<1xi64>
// CHECK:           %[[VAL_9:.*]] = arith.constant dense<[1, 4, 8]> : tensor<3xi64>
// CHECK:           %[[VAL_10:.*]] = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [0], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1]>>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<32xf32>
// CHECK:           memref.copy %[[VAL_10]], %[[VAL_11]] : memref<32xf32, strided<[1]>> to memref<32xf32>
// CHECK:           %[[VAL_12:.*]] = bufferization.to_tensor %[[VAL_11]] restrict writable : memref<32xf32>
// CHECK:           %[[VAL_13:.*]] = tensor.reshape %[[VAL_12]](%[[VAL_9]]) : (tensor<32xf32>, tensor<3xi64>) -> tensor<1x4x8xf32>
// CHECK:           %[[VAL_14:.*]] = tensor.empty() : tensor<128x4x8xf32>
// CHECK:           %[[VAL_15:.*]] = tensor.collapse_shape %[[VAL_13]] {{\[}}[0, 1], [2]] : tensor<1x4x8xf32> into tensor<4x8xf32>
// CHECK:           %[[VAL_16:.*]] = linalg.broadcast ins(%[[VAL_15]] : tensor<4x8xf32>) outs(%[[VAL_14]] : tensor<128x4x8xf32>) dimensions = [0]
// CHECK:           %[[VAL_19:.*]] = tensor.reshape %[[VAL_16]](%[[VAL_8]]) : (tensor<128x4x8xf32>, tensor<1xi64>) -> tensor<4096xf32>
// CHECK:           %[[VAL_20:.*]] = memref.reinterpret_cast [[ARG_1:%.+]] to offset: [0], sizes: [4096], strides: [1] : memref<?xf32> to memref<4096xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_19]] in writable %[[VAL_20]] : (tensor<4096xf32>, memref<4096xf32, strided<[1]>>) -> ()
// CHECK:           return


// -----
module {
  tt.func public @fn_broadcast_middle_axis(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %3 = tt.load %2 : tensor<1024x!tt.ptr<f32>>
    %4 = tt.reshape %3 : tensor<1024xf32> -> tensor<128x1x8xf32>
    %5 = tt.broadcast %4 : tensor<128x1x8xf32> -> tensor<128x4x8xf32>
    %6 = tt.reshape %5 : tensor<128x4x8xf32> -> tensor<4096xf32>
    %7 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    tt.store %9, %6 : tensor<4096x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL:   func.func @fn_broadcast_middle_axis
// CHECK:           %[[VAL_8:.*]] = arith.constant dense<4096> : tensor<1xi64>
// CHECK:           %[[VAL_9:.*]] = arith.constant dense<[128, 1, 8]> : tensor<3xi64>
// CHECK:           %[[VAL_10:.*]] = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [0], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1]>>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy %[[VAL_10]], %[[VAL_11]] : memref<1024xf32, strided<[1]>> to memref<1024xf32>
// CHECK:           %[[VAL_12:.*]] = bufferization.to_tensor %[[VAL_11]] restrict writable : memref<1024xf32>
// CHECK:           %[[VAL_13:.*]] = tensor.reshape %[[VAL_12]](%[[VAL_9]]) : (tensor<1024xf32>, tensor<3xi64>) -> tensor<128x1x8xf32>
// CHECK:           %[[VAL_14:.*]] = tensor.empty() : tensor<128x4x8xf32>
// CHECK:           %[[VAL_15:.*]] = tensor.collapse_shape %[[VAL_13]] {{\[}}[0], [1, 2]] : tensor<128x1x8xf32> into tensor<128x8xf32>
// CHECK:           %[[VAL_16:.*]] = linalg.broadcast ins(%[[VAL_15]] : tensor<128x8xf32>) outs(%[[VAL_14]] : tensor<128x4x8xf32>) dimensions = [1]
// CHECK:           %[[VAL_19:.*]] = tensor.reshape %[[VAL_16]](%[[VAL_8]]) : (tensor<128x4x8xf32>, tensor<1xi64>) -> tensor<4096xf32>
// CHECK:           %[[VAL_20:.*]] = memref.reinterpret_cast [[ARG_1:%.+]] to offset: [0], sizes: [4096], strides: [1] : memref<?xf32> to memref<4096xf32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_19]] in writable %[[VAL_20]] : (tensor<4096xf32>, memref<4096xf32, strided<[1]>>) -> ()
// CHECK:           return


// -----

module {
  // // CHECK-LABEL: func @fn_broadcast_two_axis
  tt.func public @fn_broadcast_two_axis(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    %0 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<1x!tt.ptr<f32>>, tensor<1xi32>
    %3 = tt.load %2 : tensor<1x!tt.ptr<f32>>
    %4 = tt.reshape %3 : tensor<1xf32> -> tensor<1x1xf32>
    // CHECK: %[[TENSOR:.*]] = tensor.empty() : tensor<4x8xf32>
    // CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[RESHAPE:.*]] [] : tensor<1x1xf32> into tensor<f32>
    // CHECK: %[[BRC:.*]] = linalg.broadcast ins(%[[COLLAPSED]] : tensor<f32>) outs(%[[TENSOR]] : tensor<4x8xf32>) dimensions = [0, 1]
    %5 = tt.broadcast %4 : tensor<1x1xf32> -> tensor<4x8xf32>
    %6 = tt.reshape %5 : tensor<4x8xf32> -> tensor<32xf32>
    %7 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %8 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %9, %6 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}