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
    // vector and scalar are both constant
    %4 = tt.make_range {end = 3072 : i32, start = 2048 : i32}:tensor<1024xi32>
    %c10 = arith.constant 10 : i32
    %5 = tt.splat %c10 : i32 -> tensor<1024xi32>
    %6 = arith.muli %5, %4 : tensor<1024xi32>
    //%6: splat(%c10)*range(2048, 4096);
    //%6: offset = %c10*2048, size = 1024, stride = %c10*1
    %7 = arith.addi %3, %6 : tensor<1024xi32>
    //%7: offset = %c10*2048 + %0, size = 1024, stride = %c10*1+1
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg0 offset = %c10*2048 + pid0, size = 1024, stride = %c10*1+1
    %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1024x!tt.ptr<bf16>>
    %11 = tt.addptr %10, %3 : tensor<1024x!tt.ptr<bf16>>, tensor<1024xi32>
    //source=%arg1, offset = pid0, size = 1024, stride = 1
    %16 = tt.load %9 : tensor<1024x!tt.ptr<bf16>>
    tt.store %11, %16 : tensor<1024x!tt.ptr<bf16>>
    tt.return
  }
}
// CHECK-LABEL: func.func @kernel(
// CHECK-SAME: %[[ARG_0:[A-Za-z0-9_]+]]: memref<?xi8>, %[[ARG_1:[A-Za-z0-9_]+]]: memref<?xi8>, %[[VAL_0:[A-Za-z0-9_]+]]: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %[[VAL_1:[A-Za-z0-9_]+]]: memref<?xbf16> {tt.tensor_kind = 1 : i32}
// CHECK-SAME: %[[ARG_4:[A-Za-z0-9_]+]]: i32, %[[ARG_5:[A-Za-z0-9_]+]]: i32, %[[ARG_6:[A-Za-z0-9_]+]]: i32, %[[ARG_7:[A-Za-z0-9_]+]]: i32, %[[ARG_8:.*]]: i32, %[[ARG_9:[A-Za-z0-9_]+]]: i32, %[[ARG_10:[A-Za-z0-9_]+]]: i32
// CHECK-SAME: ) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK: %[[VAL_7:[A-Za-z0-9_]+]] = arith.constant 20480 : index
// CHECK: %[[VAL_8:[A-Za-z0-9_]+]] = arith.index_cast %[[ARG_8]] : i32 to index
// CHECK: %[[VAL_9:[A-Za-z0-9_]+]] = arith.addi %[[VAL_8]], %[[VAL_7]] : index
// CHECK: %[[VAL_10:[A-Za-z0-9_]+]] = memref.reinterpret_cast %[[VAL_0]] to offset: [%[[VAL_9]]], sizes: [1024], strides: [11] : memref<?xbf16> to memref<1024xbf16, strided<[11], offset: ?>>
// CHECK: %[[VAL_12:[A-Za-z0-9_]+]] = memref.reinterpret_cast %[[VAL_1]] to offset: [%[[VAL_8]]], sizes: [1024], strides: [1] : memref<?xbf16> to memref<1024xbf16, strided<[1], offset: ?>>
// CHECK: %[[VAL_13:[A-Za-z0-9_]+]] = memref.alloc() : memref<1024xbf16>
// CHECK: memref.copy %[[VAL_10]], %[[VAL_13]] : memref<1024xbf16, strided<[11], offset: ?>> to memref<1024xbf16>
// CHECK: %[[VAL_14:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[VAL_13]] restrict writable : memref<1024xbf16>
// CHECK: bufferization.materialize_in_destination %[[VAL_14]] in writable %[[VAL_12]] : (tensor<1024xbf16>, memref<1024xbf16, strided<[1], offset: ?>>) -> ()
// CHECK: return
// CHECK: }
