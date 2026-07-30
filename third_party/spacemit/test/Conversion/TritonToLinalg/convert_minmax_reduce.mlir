// RUN: triton-shared-opt --triton-to-linalg --split-input-file %s | FileCheck %s
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

// CHECK:  func.func @minmax_sgt(%[[VAL_0:.*]]: memref<*xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = tensor.insert %c-2147483648{{.*}} into %[[VAL_9]][] : tensor<i32>
// CHECK:    %[[VAL_11:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_10]] : tensor<i32>) dimensions = [0]
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_12:.*]] = arith.maxsi %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_12:.*]] = tensor.extract %[[VAL_11]][] : tensor<i32>
// CHECK:    %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:    affine.store %[[VAL_12]], %[[VAL_13]][0] : memref<1xi32, strided<[1]>>
// CHECK:    return
// CHECK:  }

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

// CHECK:  func.func @minmax_ugt(%[[VAL_0:.*]]: memref<*xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = tensor.insert %c0{{.*}} into %[[VAL_9]][] : tensor<i32>
// CHECK:    %[[VAL_11:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_10]] : tensor<i32>) dimensions = [0]
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_12:.*]] = arith.maxui %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_12:.*]] = tensor.extract %[[VAL_11]][] : tensor<i32>
// CHECK:    %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:    affine.store %[[VAL_12]], %[[VAL_13]][0] : memref<1xi32, strided<[1]>>
// CHECK:    return
// CHECK:  }

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

// CHECK:  func.func @minmax_slt(%[[VAL_0:.*]]: memref<*xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = tensor.insert %c2147483647{{.*}} into %[[VAL_9]][] : tensor<i32>
// CHECK:    %[[VAL_11:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_10]] : tensor<i32>) dimensions = [0]
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_12:.*]] = arith.minsi %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_12:.*]] = tensor.extract %[[VAL_11]][] : tensor<i32>
// CHECK:    %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:    affine.store %[[VAL_12]], %[[VAL_13]][0] : memref<1xi32, strided<[1]>>
// CHECK:    return
// CHECK:  }

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

// CHECK:  func.func @minmax_ult(%[[VAL_0:.*]]: memref<*xi32>, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32) {
// CHECK:    %[[VAL_7:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:    %[[VAL_8:.*]] = linalg.fill ins(%c0{{.*}} : i32) outs(%[[VAL_7]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:    %[[VAL_9:.*]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:    %[[VAL_10:.*]] = tensor.insert %c-1{{.*}} into %[[VAL_9]][] : tensor<i32>
// CHECK:    %[[VAL_11:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<4096xi32>) outs(%[[VAL_10]] : tensor<i32>) dimensions = [0]
// CHECK:      (%in: i32, %init: i32) {
// CHECK:        %[[VAL_12:.*]] = arith.minui %in, %init : i32
// CHECK:        linalg.yield %[[VAL_12]] : i32
// CHECK:      }
// CHECK:    %[[VAL_12:.*]] = tensor.extract %[[VAL_11]][] : tensor<i32>
// CHECK:    %[[VAL_13:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1]>>
// CHECK:    affine.store %[[VAL_12]], %[[VAL_13]][0] : memref<1xi32, strided<[1]>>
// CHECK:    return
// CHECK:  }