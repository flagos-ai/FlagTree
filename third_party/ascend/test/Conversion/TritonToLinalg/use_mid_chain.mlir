// RUN: triton-adapter-opt --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : !tt.ptr<i32>
  )
  {
  %0 = tt.make_range {end = 768 : i32, start = 512 : i32}:tensor<256xi32>
  // offset = [512] size = 256, stride = 1
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
  // offset = [512,0], size = [256,1], stride = [1,0]
  %2 = tt.broadcast %1 : tensor<256x1xi32> -> tensor<256x128xi32>
  // offset = [512,0], size = [256,128], stride = [1,0]
  // mixed use
  %5 = tt.make_range {end = 1152 : i32, start = 1024 : i32}:tensor<128xi32>
  // offset = 1024, size = 128, stride = 1
  %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  // offset = [0,1024], size = [1,128], stride = [0,1]
  %7 = tt.broadcast %6 : tensor<1x128xi32> -> tensor<256x128xi32>
  // offset = [0,1024], size = [256,128], stride = [0,1]
  %c6 = arith.constant 6 : i32
  %splat6 = tt.splat %c6 : i32 -> tensor<256x128xi32>
  %scale7 = arith.muli %7, %splat6 : tensor<256x128xi32>
  // offset = [0,6144], size = [256,128], stride = [0,6]
  %14 = arith.addi %2, %scale7 : tensor<256x128xi32>
  // offset = [512,6144], size = [256,128], stride = [1,6]
  %17 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x128x!tt.ptr<bf16>>
  %18 = tt.addptr %17, %14 : tensor<256x128x!tt.ptr<bf16>>, tensor<256x128xi32>
  %19 = tt.load %18 : tensor<256x128x!tt.ptr<bf16>>
  tt.store %18, %19 : tensor<256x128x!tt.ptr<bf16>>
  %20 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<256x128x!tt.ptr<i32>>
  %21 = tt.addptr %20, %14 : tensor<256x128x!tt.ptr<i32>>, tensor<256x128xi32>
  tt.store %21, %2 : tensor<256x128x!tt.ptr<i32>>
  tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:    %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>,
// CHECK-SAME:    %[[VAL_0:.*]]: memref<?xbf16>, %[[VAL_1:.*]]: memref<?xbf16> {tt.tensor_kind = 2 : i32}, %[[VAL_2:.*]]: memref<?xi32> {tt.tensor_kind = 1 : i32}, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
// CHECK-DAG:     %[[VAL_7:.*]] = arith.constant 512 : i32
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<256xi32>
// CHECK:           %[[VAL_9:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%[[VAL_8]] : tensor<256xi32>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: i32):
// CHECK:             %[[VAL_11:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_12:.*]] = arith.index_cast %[[VAL_11]] : index to i32
// CHECK:             linalg.yield %[[VAL_12]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK:           %[[VAL_13:.*]] = linalg.fill ins(%[[VAL_7]] : i32) outs(%[[VAL_8]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_9]], %[[VAL_13]] : tensor<256xi32>
// CHECK:           %[[VAL_15:.*]] = tensor.empty() : tensor<256x128xi32>
// CHECK:           %[[VAL_16:.*]] = linalg.broadcast ins(%[[VAL_14]] : tensor<256xi32>) outs(%[[VAL_15]] : tensor<256x128xi32>) dimensions = [1]

// CHECK:           %[[VAL_19:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}6656], sizes: [256, 128], strides: [1, 6] : memref<?xbf16> to memref<256x128xbf16, strided<[1, 6], offset: 6656>>
// CHECK:           %[[VAL_20:.*]] = memref.alloc() : memref<256x128xbf16>
// CHECK:           memref.copy %[[VAL_19]], %[[VAL_20]] : memref<256x128xbf16, strided<[1, 6], offset: 6656>> to memref<256x128xbf16>
// CHECK:           %[[VAL_21:.*]] = bufferization.to_tensor %[[VAL_20]] restrict writable : memref<256x128xbf16>
// CHECK:           bufferization.materialize_in_destination %[[VAL_21]] in writable %[[VAL_19]]
// CHECK:           %[[VAL_22:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}6656], sizes: [256, 128], strides: [1, 6] : memref<?xi32> to memref<256x128xi32, strided<[1, 6], offset: 6656>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_23:.*]] in writable %[[VAL_22]]
// CHECK:           return
// CHECK:         }
