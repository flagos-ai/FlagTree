// RUN: triton-adapter-opt --triton-to-annotation --triton-to-unstructure  --bubble-up-operation --discrete-mask-access-conversion --triton-to-hivm "--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False" %s | FileCheck %s

module {
  tt.func @addptr_with_int_to_ptr_source(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i64>) {
    %c0_i64 = arith.constant 0 : i64
    %0 = tt.load %arg0 : !tt.ptr<i64>
    %1 = tt.int_to_ptr %0 : i64 -> !tt.ptr<i64>
    %2 = tt.addptr %1, %c0_i64 : !tt.ptr<i64>, i64
    %3 = tt.load %2 : !tt.ptr<i64>
    tt.store %arg1, %3 : !tt.ptr<i64>
    tt.return
  }
}

// CHECK-LABEL:   func.func @addptr_with_int_to_ptr_source
// CHECK:           %[[arg0:.*]]: memref<?xi64> {tt.tensor_kind = 0 : i32},
// CHECK:           %[[arg1:.*]]: memref<?xi64> {tt.tensor_kind = 1 : i32},
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_11:.*]] = memref.reinterpret_cast %[[VAL_2:.*]] to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>	
// CHECK:           %[[VAL_12:.*]] = memref.load %[[VAL_11]]{{\[}}%[[VAL_10]]] : memref<1xi64, strided<[1]>>	
// CHECK:           %[[VAL_13:.*]] = arith.constant 8 : i64	
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : index	
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : i64	
// CHECK:           %[[VAL_16:.*]] = arith.muli %[[VAL_15]], %[[VAL_13]] : i64	
// CHECK:           %[[VAL_17:.*]] = arith.addi %[[VAL_12]], %[[VAL_16]] : i64	
// CHECK:           %[[VAL_18:.*]] = hivm.hir.pointer_cast(%[[VAL_17]]) {{\[}}%[[VAL_14]]] : memref<?xi64>	
// CHECK:           annotation.mark %[[VAL_18]]	
// CHECK:           %[[VAL_19:.*]] = memref.reinterpret_cast %[[VAL_18]] to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>	
// CHECK:           %[[VAL_20:.*]] = memref.load %[[VAL_19]]{{\[}}%[[VAL_10]]] : memref<1xi64, strided<[1]>>	
// CHECK:           %[[VAL_21:.*]] = tensor.empty() : tensor<1xi64>	
// CHECK:           %[[VAL_22:.*]] = linalg.fill ins(%[[VAL_20]] : i64) outs(%[[VAL_21]] : tensor<1xi64>) -> tensor<1xi64>	
// CHECK:           %[[VAL_23:.*]] = memref.reinterpret_cast %[[VAL_3:.*]] to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>	
// CHECK:           bufferization.materialize_in_destination %[[VAL_22]] in writable %[[VAL_23]] : (tensor<1xi64>, memref<1xi64, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }