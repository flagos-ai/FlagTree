// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func public @bcast_kernel_01(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
    %6 = tt.splat %1 : i32 -> tensor<2048xi32>
    %7 = arith.addi %6, %5 : tensor<2048xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %9 = tt.addptr %8, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %10 = tt.load %9 : tensor<32x!tt.ptr<f32>>
    %11 = tt.reshape %10 : tensor<32xf32> -> tensor<1x32xf32>
    %12 = tt.broadcast %11 : tensor<1x32xf32> -> tensor<64x32xf32>
    %13 = tt.reshape %12 : tensor<64x32xf32> -> tensor<2048xf32>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    %15 = tt.addptr %14, %7 : tensor<2048x!tt.ptr<f32>>, tensor<2048xi32>
    tt.store %15, %13 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }
}


// CHECK-LABEL:  func.func @bcast_kernel_01
// CHECK:    %[[C2048_I64:.*]] = arith.constant dense<2048> : tensor<1xi64>
// CHECK:    %[[CST:.*]] = arith.constant dense<[1, 32]> : tensor<2xi64>
// CHECK:    %[[C32_I32:.*]] = arith.constant 32 : i32
// CHECK:    %[[VAR_0:.*]] = arith.muli [[PID_0:%.+]], %[[C32_I32]] : i32
// CHECK:    %[[VAR_1:.*]] = arith.index_cast %[[VAR_0]] : i32 to index
// CHECK:    %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [%[[VAR_1]]], sizes: [32], strides: [1] : memref<?xf32> to memref<32xf32, strided<[1], offset: ?>>
// CHECK:    %[[ALLOC:.*]] = memref.alloc() : memref<32xf32>
// CHECK:    memref.copy %[[REINTERPRET_CAST:.*]], %[[ALLOC]] : memref<32xf32, strided<[1], offset: ?>> to memref<32xf32>
// CHECK:    %[[VAR_2:.*]] = bufferization.to_tensor %[[ALLOC]] restrict writable : memref<32xf32>
// CHECK:    %[[RESHAPE:.*]] = tensor.reshape %[[VAR_2]](%[[CST]]) : (tensor<32xf32>, tensor<2xi64>) -> tensor<1x32xf32>
// CHECK:    %[[VAR_3:.*]] = tensor.empty() : tensor<64x32xf32>
// CHECK:    %[[VAR_4:.*]] = tensor.collapse_shape %[[RESHAPE]] {{\[}}[0, 1]] : tensor<1x32xf32> into tensor<32xf32>
// CHECK:    %[[VAR_5:.*]] = linalg.broadcast ins(%[[VAR_4]] : tensor<32xf32>) outs(%[[VAR_3]] : tensor<64x32xf32>) dimensions = [0]
// CHECK:    %[[RESHAPE_0:.*]] = tensor.reshape %[[VAR_5]](%[[C2048_I64]]) : (tensor<64x32xf32>, tensor<1xi64>) -> tensor<2048xf32>

// CHECK:    %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast [[ARG_1:%.+]] to offset: [%[[VAR_1]]], sizes: [2048], strides: [1] : memref<?xf32> to memref<2048xf32, strided<[1], offset: ?>>
// CHECK:    bufferization.materialize_in_destination %[[RESHAPE_0]] in writable %[[REINTERPRET_CAST_1]] : (tensor<2048xf32>, memref<2048xf32, strided<[1], offset: ?>>) -> ()
// CHECK:    return
