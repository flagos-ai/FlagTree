// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : i32
  )
  {
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32}:tensor<1024xi32>
    %2 = tt.splat %0 : i32 -> tensor<1024xi32>
    %3 = arith.addi %2, %1 : tensor<1024xi32>
    //%3: splat(%0) + range(0, 1024)
    //%3: offset = %0, size = 1024, stride = 1
    // vector is constant, scalar is value
    %4 = tt.make_range {end = 3072 : i32, start = 2048 : i32}:tensor<1024xi32>
    %5 = tt.splat %arg2 : i32 -> tensor<1024xi32>
    %6 = arith.muli %5, %4 : tensor<1024xi32>
    //%6: splat(%arg2)*range(2048, 3072);
    //%6: offset = %arg2*2048, size = 1024, stride = %arg2*1
    %7 = arith.addi %3, %6 : tensor<1024xi32>
    //%7: offset = %arg2*2048 + %0, size = 1024, stride = %arg2*1+1
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg0: offset = %arg2*2048 + pid0, size = 1024, stride = %arg2*1+1
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %3 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg1: offset = pid0, size = 1024, stride = 1
    %16 = tt.load %9 : tensor<1024x!tt.ptr<bf16>>
    tt.store %11, %16 : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME: %[[ARG_0:[A-Za-z0-9_]+]]: memref<?xi8>, %[[ARG_1:[A-Za-z0-9_]+]]: memref<?xi8>, %[[VAL_0:[A-Za-z0-9_]+]]: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %[[VAL_1:[A-Za-z0-9_]+]]: memref<?xbf16> {tt.tensor_kind = 1 : i32}
// CHECK-SAME: %[[ARG_4:[A-Za-z0-9_]+]]: i32, %[[ARG_5:[A-Za-z0-9_]+]]: i32, %[[ARG_6:[A-Za-z0-9_]+]]: i32, %[[ARG_7:[A-Za-z0-9_]+]]: i32, %[[ARG_8:.*]]: i32, %[[ARG_9:[A-Za-z0-9_]+]]: i32, %[[ARG_10:[A-Za-z0-9_]+]]: i32
// CHECK-SAME: ) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK-DAG:           %[[VAL_8:.*]] = arith.constant 2048 : index
// CHECK:           %[[VAL_9:.*]] = arith.index_cast %[[ARG_8]] : i32 to index
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[ARG_4]] : i32 to index
// CHECK:           %[[VAL_11:.*]] = arith.muli %[[VAL_10]], %[[VAL_8]] : index
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_9]], %[[VAL_11]] : index
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_10]], %[[VAL_6]] : index
// CHECK:           %[[VAL_15:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_13]]], sizes: [1024], strides: {{\[}}%[[VAL_14]]] : memref<?xbf16> to memref<1024xbf16, strided<[?], offset: ?>>
// CHECK:           %[[VAL_17:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_9]]], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
// CHECK:           %[[VAL_18:.*]] = memref.alloc() : memref<1024xbf16>
// CHECK:           memref.copy %[[VAL_15]], %[[VAL_18]] : memref<1024xbf16, strided<[?], offset: ?>> to memref<1024xbf16>
// CHECK:           %[[VAL_19:.*]] = bufferization.to_tensor %[[VAL_18]] restrict writable : memref<1024xbf16>
// CHECK:           bufferization.materialize_in_destination %[[VAL_19]] in writable %[[VAL_17]]
// CHECK:           return
// CHECK:         }
