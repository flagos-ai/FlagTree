// RUN: triton-adapter-opt --triton-to-linalg --split-input-file %s | FileCheck %s
module {
  tt.func public @minmax_sgt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi sgt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_sgt
// CHECK:    %[[VAL_c0:.*]] = arith.constant 0 : i32
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = tensor.empty() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_9]] : tensor<i32>) dimensions = [0] {reduce_mode = "max_with_index"}
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_11:.*]] = arith.cmpi sgt, %in, %init : i32
// CHECK:        %[[VAL_12:.*]] = arith.select %[[VAL_11]], %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_13:.*]] = tensor.extract %[[VAL_10]][] : tensor<i32>
// CHECK:    %[[VAL_14:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:    %[[VAL_15:.*]] = linalg.fill ins(%[[VAL_13]] : i32) outs(%[[VAL_14]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:    %reinterpret_cast = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
// CHECK:    bufferization.materialize_in_destination %[[VAL_15]] in writable %reinterpret_cast : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
// CHECK:    return


// -----

module {
  tt.func public @minmax_ugt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi ugt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_ugt
// CHECK:    %[[VAL_c0:.*]] = arith.constant 0 : i32
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = tensor.empty() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_9]] : tensor<i32>) dimensions = [0] {reduce_mode = "max_with_index"}
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_11:.*]] = arith.cmpi ugt, %in, %init : i32
// CHECK:        %[[VAL_12:.*]] = arith.select %[[VAL_11]], %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_13:.*]] = tensor.extract %[[VAL_10]][] : tensor<i32>
// CHECK:    %[[VAL_14:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:    %[[VAL_15:.*]] = linalg.fill ins(%[[VAL_13]] : i32) outs(%[[VAL_14]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:    %reinterpret_cast = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
// CHECK:    bufferization.materialize_in_destination %[[VAL_15]] in writable %reinterpret_cast : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
// CHECK:    return

// -----

module {
  tt.func public @minmax_slt(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi slt, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_slt
// CHECK:    %[[VAL_c0:.*]] = arith.constant 0 : i32
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = tensor.empty() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_9]] : tensor<i32>) dimensions = [0] {reduce_mode = "min_with_index"}
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_11:.*]] = arith.cmpi slt, %in, %init : i32
// CHECK:        %[[VAL_12:.*]] = arith.select %[[VAL_11]], %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_13:.*]] = tensor.extract %[[VAL_10]][] : tensor<i32>
// CHECK:    %[[VAL_14:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:    %[[VAL_15:.*]] = linalg.fill ins(%[[VAL_13]] : i32) outs(%[[VAL_14]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:    %reinterpret_cast = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
// CHECK:    bufferization.materialize_in_destination %[[VAL_15]] in writable %reinterpret_cast : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
// CHECK:    return

// -----

module {
  tt.func public @minmax_ult(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.cmpi ult, %arg14, %arg15 : i32
      %70 = arith.select %69, %arg14, %arg15 : i32
      tt.reduce.return %70 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK:  func.func @minmax_ult
// CHECK:    %[[VAL_c0:.*]] = arith.constant 0 : i32
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = tensor.empty() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_9]] : tensor<i32>) dimensions = [0] {reduce_mode = "min_with_index"}
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_11:.*]] = arith.cmpi ult, %in, %init : i32
// CHECK:        %[[VAL_12:.*]] = arith.select %[[VAL_11]], %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_13:.*]] = tensor.extract %[[VAL_10]][] : tensor<i32>
// CHECK:    %[[VAL_14:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:    %[[VAL_15:.*]] = linalg.fill ins(%[[VAL_13]] : i32) outs(%[[VAL_14]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:    %reinterpret_cast = memref.reinterpret_cast [[ARG_0:%.+]] to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
// CHECK:    bufferization.materialize_in_destination %[[VAL_15]] in writable %reinterpret_cast : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
// CHECK:    return
