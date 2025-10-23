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
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<1xi64>
// CHECK:           %[[VAL_12:.*]] = memref.subview %[[VAL_2:.*]][0] [1] [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
// CHECK:           memref.copy %[[VAL_12]], %[[VAL_11]] : memref<1xi64, strided<[1]>> to memref<1xi64>
// CHECK:           %[[VAL_13:.*]] = bufferization.to_tensor %[[VAL_11]] restrict writable : memref<1xi64>
// CHECK:           %[[VAL_14:.*]] = tensor.extract %[[VAL_13]]{{\[}}%[[VAL_10]]] : tensor<1xi64>
// CHECK:           %[[VAL_15:.*]] = arith.constant 8 : i64
// CHECK:           %[[VAL_16:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_17:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_18:.*]] = arith.muli %[[VAL_17]], %[[VAL_15]] : i64
// CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_14]], %[[VAL_18]] : i64
// CHECK:           %[[VAL_20:.*]] = hivm.hir.pointer_cast(%[[VAL_19]]) {{\[}}%[[VAL_16]]] : memref<?xi64>
// CHECK:           annotation.mark %[[VAL_20]] {address_space = #hivm.address_space<gm>} : memref<?xi64>
// CHECK:           %[[VAL_21:.*]] = memref.reinterpret_cast %[[VAL_20]] to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
// CHECK:           %[[VAL_22:.*]] = memref.alloc() : memref<1xi64>
// CHECK:           memref.copy %[[VAL_21]], %[[VAL_22]] : memref<1xi64, strided<[1]>> to memref<1xi64>
// CHECK:           %[[VAL_23:.*]] = bufferization.to_tensor %[[VAL_22]] restrict writable : memref<1xi64>
// CHECK:           %[[VAL_24:.*]] = tensor.extract %[[VAL_23]]{{\[}}%[[VAL_10]]] : tensor<1xi64>
// CHECK:           %[[VAL_25:.*]] = tensor.empty() : tensor<1xi64>
// CHECK:           %[[VAL_26:.*]] = linalg.fill ins(%[[VAL_24]] : i64) outs(%[[VAL_25]] : tensor<1xi64>) -> tensor<1xi64>
// CHECK:           %[[VAL_27:.*]] = memref.reinterpret_cast %[[VAL_3:.*]] to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_26]] in writable %[[VAL_27]] : (tensor<1xi64>, memref<1xi64, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }