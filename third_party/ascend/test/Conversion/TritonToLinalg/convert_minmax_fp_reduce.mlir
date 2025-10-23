// RUN: triton-adapter-opt --triton-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @maxnumf(%arg0: !tt.ptr<f32>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4096xf32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: f32, %arg15: f32):
      %69 = arith.maxnumf %arg14, %arg15 : f32
      tt.reduce.return %69 : f32
    }) {axis = 0 : i32} : (tensor<4096xf32>) -> f32
    tt.store %arg0, %63 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL: func.func @maxnumf
// CHECK:  %[[CST:.*]] = arith.constant 0xFF800000 : f32
// CHECK:  %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<4096xf32>
// CHECK:  %[[VAL_1:.*]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[VAL_0]] : tensor<4096xf32>) -> tensor<4096xf32>
// CHECK:  %[[VAL_2:.*]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:  %[[VAL_3:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[VAL_2]] : tensor<f32>) -> tensor<f32>
// CHECK:  %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_1]] : tensor<4096xf32>) outs(%[[VAL_3]] : tensor<f32>) dimensions = [0]
// CHECK:    (%in: f32, %init: f32) {
// CHECK:      %[[VAL_5:.*]] = arith.maxnumf %in, %init : f32
// CHECK:      linalg.yield %[[VAL_5]] : f32
// CHECK:    }
// CHECK:  %[[VAL_6:.*]] = tensor.extract %[[VAL_4]][] : tensor<f32>
// CHECK:  %[[VAL_7:.*]] = tensor.empty() : tensor<1xf32>
// CHECK:  %[[VAL_8:.*]] = linalg.fill ins(%extracted : f32) outs(%[[VAL_7]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK-DAG:  %[[VAL_9:.*]] = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
// CHECK:  bufferization.materialize_in_destination %[[VAL_8]] in writable %[[VAL_9]] : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
// CHECK:  return


// -----


module {
  tt.func public @minnumf(%arg0: !tt.ptr<f32>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<4096xf32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: f32, %arg15: f32):
      %69 = arith.minnumf %arg14, %arg15 : f32
      tt.reduce.return %69 : f32
    }) {axis = 0 : i32} : (tensor<4096xf32>) -> f32
    tt.store %arg0, %63 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL:   func.func @minnumf
// CHECK:  %[[CST:.*]] = arith.constant 0x7F800000 : f32
// CHECK:  %[[CST_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:  %[[VAL_0:.*]] = tensor.empty() : tensor<4096xf32>
// CHECK:  %[[VAL_1:.*]] = linalg.fill ins(%[[CST_0]] : f32) outs(%[[VAL_0]] : tensor<4096xf32>) -> tensor<4096xf32>
// CHECK:  %[[VAL_2:.*]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:  %[[VAL_3:.*]] = linalg.fill ins(%[[CST]] : f32) outs(%[[VAL_2]] : tensor<f32>) -> tensor<f32>
// CHECK:  %[[VAL_4:.*]] = linalg.reduce ins(%[[VAL_1]] : tensor<4096xf32>) outs(%[[VAL_3]] : tensor<f32>) dimensions = [0]
// CHECK:    (%in: f32, %init: f32) {
// CHECK:      %[[VAL_5:.*]] = arith.minnumf %in, %init : f32
// CHECK:      linalg.yield %[[VAL_5]] : f32
// CHECK:    }
// CHECK:  %[[VAL_6:.*]] = tensor.extract %[[VAL_4]][] : tensor<f32>
// CHECK:  %[[VAL_7:.*]] = tensor.empty() : tensor<1xf32>
// CHECK:  %[[VAL_8:.*]] = linalg.fill ins(%extracted : f32) outs(%[[VAL_7]] : tensor<1xf32>) -> tensor<1xf32>
// CHECK:  %[[VAL_9:.*]] = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [0], sizes: [1], strides: [1] : memref<?xf32> to memref<1xf32, strided<[1]>>
// CHECK:  bufferization.materialize_in_destination %[[VAL_8]] in writable %[[VAL_9]] : (tensor<1xf32>, memref<1xf32, strided<[1]>>) -> ()
// CHECK:  return

