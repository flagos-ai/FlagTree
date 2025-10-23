// RUN: triton-adapter-opt --triton-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func @kernel_low_mask(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  )
  {
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %ldptr = tt.addptr %0, %2 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %stptr = tt.addptr %1, %2 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %nans = arith.constant dense<0xFF80> : tensor<128xbf16>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %mask = arith.cmpi slt, %2, %5 : tensor<128xi32>
    %buff = tt.load %ldptr, %mask, %nans : tensor<128x!tt.ptr<bf16>>
    tt.store %stptr, %buff, %mask : tensor<128x!tt.ptr<bf16>>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel_low_mask(
// CHECK-SAME:          %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, 
// CHECK-SAME:         %[[VAL_0:.*]]: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %[[VAL_1:.*]]: memref<?xbf16> {tt.tensor_kind = 1 : i32}, %[[VAL_2:.*]]: i32, %[[ARG_3:.*]]: i32, %[[ARG_4:.*]]: i32, %[[ARG_5:.*]]: i32, %[[ARG_6:.*]]: i32, %[[ARG_7:.*]]: i32, %[[ARG_8:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} { 

// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 0xFF80 : bf16
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_8:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [128], strides: [1] : memref<?xbf16> to memref<128xbf16, strided<[1]>>
// CHECK:           %[[VAL_9:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [128], strides: [1] : memref<?xbf16> to memref<128xbf16, strided<[1]>>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() : memref<128xbf16>
// CHECK:           %[[VAL_11:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_12_0:.*]] = arith.maxsi %[[VAL_11]], %[[VAL_5]] : index
// CHECK:           %[[VAL_12:.*]] = arith.minsi %[[VAL_12_0]], %[[VAL_7]] : index
// CHECK:           %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_7]] : index
// CHECK:           scf.if %[[VAL_15]] {
// CHECK:             linalg.fill ins(%[[VAL_6]] : bf16) outs(%[[VAL_10]] : memref<128xbf16>)
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = memref.subview %[[VAL_8]][0] {{\[}}%[[VAL_12]]] [1] : memref<128xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
// CHECK:           %[[VAL_14:.*]] = memref.subview %[[VAL_10]][0] {{\[}}%[[VAL_12]]] [1] : memref<128xbf16> to memref<?xbf16, strided<[1]>>
// CHECK:           memref.copy %[[VAL_13]], %[[VAL_14]] : memref<?xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
// CHECK:           %[[VAL_16:.*]] = bufferization.to_tensor %[[VAL_10]] restrict writable : memref<128xbf16>

// CHECK:           %[[VAL_19:.*]] = tensor.extract_slice %[[VAL_16]][0] {{\[}}%[[VAL_12]]] [1] : tensor<128xbf16> to tensor<?xbf16>
// CHECK:           %[[VAL_20:.*]] = memref.subview %[[VAL_9]][0] {{\[}}%[[VAL_12]]] [1] : memref<128xbf16, strided<[1]>> to memref<?xbf16, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_19]] in writable %[[VAL_20]]
// CHECK:           return
// CHECK:         }

// -----

module {
  tt.func @kernel_high_mask(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  )
  {
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %ldptr = tt.addptr %0, %2 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %stptr = tt.addptr %1, %2 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %nans = arith.constant dense<0xFF80> : tensor<128xbf16>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %mask = arith.cmpi sge, %2, %5 : tensor<128xi32>
    %buff = tt.load %ldptr, %mask, %nans : tensor<128x!tt.ptr<bf16>>
    tt.store %stptr, %buff, %mask : tensor<128x!tt.ptr<bf16>>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel_high_mask(
// CHECK-SAME:          %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, %[[PA_0:.*]]: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %[[PA_1:.*]]: memref<?xbf16> {tt.tensor_kind = 1 : i32}, %[[PA_2:.*]]: i32, %[[ARG_3:.*]]: i32, %[[ARG_4:.*]]: i32, %[[ARG_5:.*]]: i32, %[[ARG_6:.*]]: i32, %[[ARG_7:.*]]: i32, %[[ARG_8:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} { 
// CHECK-DAG:           %[[CONST_NAN:.*]] = arith.constant 0xFF80 : bf16
// CHECK-DAG:           %[[CONST_128:.*]] = arith.constant 128 : index
// CHECK-DAG:           %[[CONST_0:.*]] = arith.constant 0 : index
// CHECK:           %[[CAST_IN:.*]] = memref.reinterpret_cast %[[PA_0]] to offset: [0], sizes: [128], strides: [1] : memref<?xbf16> to memref<128xbf16, strided<[1]>>
// CHECK:           %[[CAST_OUT:.*]] = memref.reinterpret_cast %[[PA_1]] to offset: [0], sizes: [128], strides: [1] : memref<?xbf16> to memref<128xbf16, strided<[1]>>
// CHECK:           %[[BUFF:.*]] = memref.alloc() : memref<128xbf16>
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[PA_2]] : i32 to index
// CHECK:           %[[VAL_1_0:.*]] = arith.maxsi %[[VAL_0]], %[[CONST_0]] : index
// CHECK:           %[[VAL_1:.*]] = arith.minsi %[[VAL_1_0]], %[[CONST_128]] : index
// CHECK:           %[[VAL_2:.*]] = arith.subi %[[CONST_128]], %[[VAL_1]] : index
// CHECK:           %[[VAL_3:.*]] = arith.cmpi slt, %[[VAL_2]], %[[CONST_128]] : index
// CHECK:           scf.if %[[VAL_3]] {
// CHECK:             linalg.fill ins(%[[CONST_NAN]] : bf16) outs(%[[BUFF]] : memref<128xbf16>)
// CHECK:           }
// CHECK:           %[[SUB_IN:.*]] = memref.subview %[[CAST_IN]]{{\[}}%[[VAL_1]]] {{\[}}%[[VAL_2]]] [1] : memref<128xbf16, strided<[1]>> to memref<?xbf16, strided<[1], offset: ?>>
// CHECK:           %[[SUB_BUFF:.*]] = memref.subview %[[BUFF]]{{\[}}%[[VAL_1]]] {{\[}}%[[VAL_2]]] [1] : memref<128xbf16> to memref<?xbf16, strided<[1], offset: ?>>
// CHECK:           memref.copy %[[SUB_IN]], %[[SUB_BUFF]] : memref<?xbf16, strided<[1], offset: ?>> to memref<?xbf16, strided<[1], offset: ?>>
// CHECK:           %[[VAL_4:.*]] = bufferization.to_tensor %[[BUFF]] restrict writable : memref<128xbf16>
// CHECK:           %[[SLICE_BUFF:.*]] = tensor.extract_slice %[[VAL_4]]{{\[}}%[[VAL_1]]] {{\[}}%[[VAL_2]]] [1] : tensor<128xbf16> to tensor<?xbf16>
// CHECK:           %[[SUB_OUT:.*]] = memref.subview %[[CAST_OUT]]{{\[}}%[[VAL_1]]] {{\[}}%[[VAL_2]]] [1] : memref<128xbf16, strided<[1]>> to memref<?xbf16, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination %[[SLICE_BUFF]] in writable %[[SUB_OUT]]
// CHECK:           return
// CHECK:         }
