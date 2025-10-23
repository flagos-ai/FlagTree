// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(%afloat : !tt.ptr<bf16>, %res : !tt.ptr<bf16>)
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %afm = tt.load %2 : tensor<128x!tt.ptr<bf16>>
    %3 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 0 : i32} : (tensor<128xbf16>) -> bf16
    tt.store %res, %3 : !tt.ptr<bf16>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:      %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>,
// CHECK-SAME:      %[[VAL_0:.*]]: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %[[VAL_1:.*]]: memref<?xbf16> {tt.tensor_kind = 1 : i32}, %[[ARG_4:.*]]: i32, %[[ARG_5:.*]]: i32, %[[ARG_6:.*]]: i32, %[[ARG_7:.*]]: i32, %[[ARG_8:.*]]: i32, %[[ARG_9:.*]]: i32) 
// CHECK-SAME:      attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK:           %[[VAL_6:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [128], strides: [1] : memref<?xbf16> to memref<128xbf16, strided<[1]>>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<128xbf16>
// CHECK:           memref.copy %[[VAL_6]], %[[VAL_7]] : memref<128xbf16, strided<[1]>> to memref<128xbf16>
// CHECK:           %[[VAL_8:.*]] = bufferization.to_tensor %[[VAL_7]] restrict writable : memref<128xbf16>
// CHECK:           %[[VAL_9:.*]] = bufferization.alloc_tensor() : tensor<bf16>
// CHECK:           %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_5]] : bf16) outs(%[[VAL_9]] : tensor<bf16>) -> tensor<bf16>
// CHECK:           %[[VAL_11:.*]] = linalg.reduce ins(%[[VAL_8]] : tensor<128xbf16>) outs(%[[VAL_10]] : tensor<bf16>) dimensions = [0]
// CHECK:             (%[[VAL_12:.*]]: bf16, %[[VAL_13:.*]]: bf16) {              
// CHECK:               %[[VAL_15:.*]] = arith.addf %[[VAL_12]], %[[VAL_13]] : bf16
// CHECK:               linalg.yield %[[VAL_15]] : bf16
// CHECK:             }
// CHECK:           %[[VAL_16:.*]] = tensor.extract %[[VAL_11]][] : tensor<bf16>
// CHECK:           %[[VAL_18:.*]] = tensor.empty() : tensor<1xbf16>
// CHECK:           %[[VAL_19:.*]] = linalg.fill ins(%[[VAL_16]] : bf16) outs(%[[VAL_18]] : tensor<1xbf16>) -> tensor<1xbf16>
// CHECK:           %[[VAL_20:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_19]] in writable %[[VAL_20]] : (tensor<1xbf16>, memref<1xbf16, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }