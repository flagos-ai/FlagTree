// RUN: triton-adapter-opt --discrete-mask-access-conversion --triton-to-annotation --triton-to-unstructure --triton-to-hivm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False' %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>
  )
  {
  %0 = tt.make_range {end = 768 : i32, start = 512 : i32}:tensor<256xi32>
  // offset = [512] size = 256, stride = 1
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
  // offset = [512,0], size = [256,1], stride = [1,0]
  %2 = tt.broadcast %1 : tensor<256x1xi32> -> tensor<256x128xi32>
  // offset = [512,0], size = [256,128], stride = [1,0]
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
  // mixed use
  %17 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x128x!tt.ptr<bf16>>
  %18 = tt.addptr %17, %14 : tensor<256x128x!tt.ptr<bf16>>, tensor<256x128xi32>
  %19 = tt.load %18 : tensor<256x128x!tt.ptr<bf16>>
  tt.store %18, %19 : tensor<256x128x!tt.ptr<bf16>>
  %20 = arith.sitofp %14 : tensor<256x128xi32> to tensor<256x128xbf16>
  tt.store %18, %20 : tensor<256x128x!tt.ptr<bf16>>
  tt.return
  }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:    %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>,
// CHECK-SAME:          %[[VAL_0:.*]]: memref<?xbf16>,  %[[VAL_1:.*]]: memref<?xbf16> {tt.tensor_kind = 2 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv"} {
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 1024 : i32
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 512 : i32
// CHECK-DAG:           %[[VAL_7:.*]] = arith.constant 6 : i32
// CHECK:           %[[VAL_30:.*]] = tensor.empty() : tensor<256x128xi32>
// CHECK:           %[[VAL_31:.*]] = linalg.fill ins(%[[VAL_7]] : i32) outs(%[[VAL_30]] : tensor<256x128xi32>) -> tensor<256x128xi32>
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<256xi32>
// CHECK:           %[[VAL_9:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%[[VAL_8]] : tensor<256xi32>) {
// CHECK:           ^bb0(%[[VAL_10:.*]]: i32):
// CHECK:             %[[VAL_11:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_12:.*]] = arith.index_cast %[[VAL_11]] : index to i32
// CHECK:             linalg.yield %[[VAL_12]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK:           %[[VAL_13:.*]] = linalg.fill ins(%[[VAL_6]] : i32) outs(%[[VAL_8]] : tensor<256xi32>) -> tensor<256xi32>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_9]], %[[VAL_13]] : tensor<256xi32>
// CHECK:           %[[VAL_15:.*]] = linalg.broadcast ins(%[[VAL_14]] : tensor<256xi32>) outs(%[[VAL_30]] : tensor<256x128xi32>) dimensions = [1] 
// CHECK:           %[[VAL_16:.*]] = tensor.empty() : tensor<128xi32>

// CHECK:           %[[VAL_20:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%[[VAL_16]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_21:.*]]: i32):
// CHECK:             %[[VAL_22:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_23:.*]] = arith.index_cast %[[VAL_22]] : index to i32
// CHECK:             linalg.yield %[[VAL_23]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[VAL_24:.*]] = linalg.fill ins(%[[VAL_5]] : i32) outs(%[[VAL_16]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_20]], %[[VAL_24]] : tensor<128xi32>
// CHECK:           %[[VAL_26:.*]] = linalg.broadcast ins(%[[VAL_25]] : tensor<128xi32>) outs(%[[VAL_30]] : tensor<256x128xi32>) dimensions = [0]
// CHECK:           %[[VAL_32:.*]] = arith.muli %[[VAL_26]], %[[VAL_31]] : tensor<256x128xi32>
// CHECK:           %[[VAL_38:.*]] = arith.addi %[[VAL_15]], %[[VAL_32]] : tensor<256x128xi32>
// CHECK:           %[[VAL_45:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}6656], sizes: [256, 128], strides: [1, 6] : memref<?xbf16> to memref<256x128xbf16, strided<[1, 6], offset: 6656>>
// CHECK:           %[[VAL_46:.*]] = memref.alloc() : memref<256x128xbf16>
// CHECK:           memref.copy %[[VAL_45]], %[[VAL_46]] : memref<256x128xbf16, strided<[1, 6], offset: 6656>> to memref<256x128xbf16>
// CHECK:           %[[VAL_47:.*]] = bufferization.to_tensor %[[VAL_46]] restrict writable : memref<256x128xbf16>
// CHECK:           bufferization.materialize_in_destination %[[VAL_47]] in writable %[[VAL_45]] : (tensor<256x128xbf16>, memref<256x128xbf16, strided<[1, 6], offset: 6656>>) -> ()
// CHECK:           %[[VAL_48:.*]] = arith.sitofp %[[VAL_38]] : tensor<256x128xi32> to tensor<256x128xbf16>
// CHECK:           bufferization.materialize_in_destination %[[VAL_48]] in writable %[[VAL_45]] : (tensor<256x128xbf16>, memref<256x128xbf16, strided<[1, 6], offset: 6656>>) -> ()
// CHECK:           return
// CHECK:         }
