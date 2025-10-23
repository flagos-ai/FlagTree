// RUN: triton-adapter-opt --triton-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @addi(%arg0: !tt.ptr<i32>) {
    %cst_0 = arith.constant dense<0> : tensor<4096xi32>
    %63 = "tt.reduce"(%cst_0) ({
    ^bb0(%arg14: i32, %arg15: i32):
      %69 = arith.addi %arg14, %arg15 : i32
      tt.reduce.return %69 : i32
    }) {axis = 0 : i32} : (tensor<4096xi32>) -> i32
    tt.store %arg0, %63 : !tt.ptr<i32>
    tt.return
  }
}


// CHECK:    %[[VAL_1:.*]] = tensor.extract %reduced[] : tensor<i32>
// CHECK:    %[[VAL_2:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:    %[[VAL_3:.*]] = linalg.fill ins(%[[VAL_1]] : i32) outs(%[[VAL_2]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:    %[[VAL_4:.*]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
// CHECK:    bufferization.materialize_in_destination %[[VAL_3]] in writable %[[VAL_4]] : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
// CHECK:    return
