// RUN: triton-adapter-opt --triton-to-annotation --triton-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  )
  {
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
  // offset = 0, size = 4, stride = 1
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  // offset = [0,0], size = [4,1], stride = [1,0]
  %2 = tt.broadcast %1 : tensor<4x1xi32> -> tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [1,0]
  %arg2splat = tt.splat %arg2 : i32 -> tensor<4x256xi32>
  %offset2 = arith.addi %2, %arg2splat : tensor<4x256xi32>
  // offset = [%arg2,0], size = [4,256], stride = [1,0]
  %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
  // offset = 0, size = 256, stride = 1
  %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
  // offset = [0,0], size = [1,256], stride = [0,1]
  %5 = tt.broadcast %4 : tensor<1x256xi32> -> tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [0,1]
  %c6 = arith.constant 6 : i32
  %splat6 = tt.splat %c6 : i32 -> tensor<4x256xi32>
  %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [0,6]
  %7 = arith.addi %offset2, %scale5: tensor<4x256xi32>
  // offset = [%arg2, 0], size = [4, 256], stride = [1, 6]
  %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
  %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
  // source: arg0, offset = [%arg2, 0], size = [4, 256], stride = [1, 6]
  %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
  %11 = tt.addptr %10, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
  // source: arg1, offset = [%arg2, 0], size = [4, 256], stride = [1, 6]
  %12 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
  tt.store %11, %12 : tensor<4x256x!tt.ptr<bf16>>
  tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:       %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, %[[VAL_0:.*]]: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %[[VAL_1:.*]]: memref<?xbf16> {tt.tensor_kind = 1 : i32}, 
// CHECK-SAME:       %[[VAL_2:.*]]: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:           %[[VAL_7:.*]] = arith.index_cast %[[VAL_2]] : i32 to index
// CHECK:           %[[VAL_8:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_7]]], sizes: [4, 256], strides: [1, 6] : memref<?xbf16> to memref<4x256xbf16, strided<[1, 6], offset: ?>>
// CHECK:           %[[VAL_10:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_7]]], sizes: [4, 256], strides: [1, 6] : memref<?xbf16> to memref<4x256xbf16, strided<[1, 6], offset: ?>>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<4x256xbf16>
// CHECK:           memref.copy %[[VAL_8]], %[[VAL_11]] : memref<4x256xbf16, strided<[1, 6], offset: ?>> to memref<4x256xbf16>
// CHECK:           %[[VAL_12:.*]] = bufferization.to_tensor %[[VAL_11]] restrict writable : memref<4x256xbf16>
// CHECK:           bufferization.materialize_in_destination %[[VAL_12]] in writable %[[VAL_10]]
// CHECK:           return
// CHECK:         }
