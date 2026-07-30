// RUN: triton-shared-opt  --structured-to-memref --canonicalize --cse %s | FileCheck %s

// CHECK-LABEL:   tt.func public @row_gather(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:                               %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant dense<5> : tensor<8xi32>
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<3> : tensor<8xi32>
// CHECK:           %[[VAL_7:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant dense<8> : tensor<8xi32>
// CHECK:           %[[VAL_9:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[VAL_10:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[VAL_11:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[VAL_12:.*]] = memref.reinterpret_cast %[[VAL_11]] to offset: [0], sizes: [8], strides: [1] : memref<*xi32> to memref<8xi32, strided<[1]>>
// CHECK:           %[[VAL_13:.*]] = memref.alloc() : memref<8xi32>
// CHECK:           memref.copy %[[VAL_12]], %[[VAL_13]] : memref<8xi32, strided<[1]>> to memref<8xi32>
// CHECK:           %[[VAL_14:.*]] = bufferization.to_tensor %[[VAL_13]] restrict writable : memref<8xi32> to tensor<8xi32>
// CHECK:           %[[VAL_15:.*]] = arith.muli %[[VAL_14]], %[[VAL_6]] : tensor<8xi32>
// CHECK:           %[[VAL_16:.*]] = arith.remsi %[[VAL_15]], %[[VAL_5]] : tensor<8xi32>
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_16]], %[[VAL_8]] : tensor<8xi32>
// CHECK:           %[[VAL_18:.*]] = memref.alloc() : memref<8x8xf32>
// CHECK:           scf.for %[[VAL_19:.*]] = %[[VAL_4]] to %[[VAL_7]] step %[[VAL_3]] {
// CHECK:             %[[VAL_20:.*]] = tensor.extract %[[VAL_17]]{{\[}}%[[VAL_19]]] : tensor<8xi32>
// CHECK:             %[[VAL_21:.*]] = arith.index_cast %[[VAL_20]] : i32 to index
// CHECK:             %[[VAL_22:.*]] = memref.reinterpret_cast %[[VAL_10]] to offset: {{\[}}%[[VAL_21]]], sizes: [1, 8], strides: [1, 1] : memref<*xf32> to memref<1x8xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_23:.*]] = memref.subview %[[VAL_18]]{{\[}}%[[VAL_19]], 0] [1, 8] [1, 1] : memref<8x8xf32> to memref<1x8xf32, strided<[8, 1], offset: ?>>
// CHECK:             memref.copy %[[VAL_22]], %[[VAL_23]] : memref<1x8xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[8, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[VAL_24:.*]] = bufferization.to_tensor %[[VAL_18]] restrict writable : memref<8x8xf32> to tensor<8x8xf32>
// CHECK:           %[[VAL_25:.*]] = memref.reinterpret_cast %[[VAL_9]] to offset: [0], sizes: [8, 8], strides: [8, 1] : memref<*xf32> to memref<8x8xf32, strided<[8, 1]>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_24]] in writable %[[VAL_25]] : (tensor<8x8xf32>, memref<8x8xf32, strided<[8, 1]>>) -> ()


module {
  tt.func public @row_gather(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<8xi32>
    %c8 = arith.constant 8 : index
    %cst_0 = arith.constant dense<3> : tensor<8xi32>
    %cst_1 = arith.constant dense<5> : tensor<8xi32>
    %0 = tts.make_tptr %arg1 to sizes: [8], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<8x!tt.ptr<i32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<8x!tt.ptr<i32>>) -> tensor<8xi32>
    %2 = arith.muli %1, %cst_0 : tensor<8xi32>
    %3 = arith.remsi %2, %cst_1 : tensor<8xi32>
    %4 = arith.muli %3, %cst : tensor<8xi32>
    %5 = tts.make_gather_scatter_tptr %arg0 to sizes: [8, 8] gather_scatter_dim: 0 gather_scatter_offset: %4, strides: [1, 1], offsets: [0, 0] : tensor<8xi32> <f32> to !tt.ptr<tensor<8x8xf32>>
    %6 = "tts.load"(%5) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<8x8xf32>>) -> tensor<8x8xf32>
    %7 = tts.make_tptr %arg2 to sizes: [8, 8], strides: [%c8, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<8x8x!tt.ptr<f32>>
    "tts.store"(%7, %6) <{static_mask_dims = array<i64>}> : (tensor<8x8x!tt.ptr<f32>>, tensor<8x8xf32>) -> ()
    tt.return
  }
}