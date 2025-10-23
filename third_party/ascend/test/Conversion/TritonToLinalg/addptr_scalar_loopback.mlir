// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  ) {
    %0 = tt.addptr %arg0, %arg2 : !tt.ptr<bf16>, i32
    %1 = tt.addptr %arg1, %arg2 : !tt.ptr<bf16>, i32
    %10 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: !tt.ptr<bf16>
    tt.store %1, %10 : !tt.ptr<bf16>
    tt.return
  }
}

// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %[[VAL_0:.*]]: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %[[VAL_1:.*]]: memref<?xbf16> {tt.tensor_kind = 1 : i32}, %[[VAL_2:.*]]: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32)  attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK-DAG: %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:     %[[VAL_4:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:     %[[VAL_5:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_4]]], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
// CHECK:     %[[VAL_6:.*]] = memref.alloc() : memref<1xbf16>
// CHECK:     memref.copy %[[VAL_5]], %[[VAL_6]] : memref<1xbf16, strided<[1], offset: ?>> to memref<1xbf16>
// CHECK:     %[[VAL_7:.*]] = bufferization.to_tensor %[[VAL_6]] restrict writable : memref<1xbf16>
// CHECK:     %[[VAL_8:.*]] = tensor.extract %[[VAL_7]]{{\[}}%[[VAL_3]]] : tensor<1xbf16>
// CHECK:     %[[VAL_9:.*]] = tensor.empty() : tensor<1xbf16>
// CHECK:     %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_8]] : bf16) outs(%[[VAL_9]] : tensor<1xbf16>) -> tensor<1xbf16>
// CHECK:     %[[VAL_11:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [%[[VAL_4]]], sizes: [1], strides: [1] : memref<?xbf16> to memref<1xbf16, strided<[1], offset: ?>>
// CHECK:     bufferization.materialize_in_destination %[[VAL_10]] in writable %[[VAL_11]] : (tensor<1xbf16>, memref<1xbf16, strided<[1], offset: ?>>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
