// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<bf16>,
        %res : !tt.ptr<bf16>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : i32 -> tensor<512xi32>
    %ws = arith.muli %ct256, %0 : tensor<512xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
    %moff = tt.broadcast %1 : tensor<512x1xi32> -> tensor<512x256xi32>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %koff = tt.broadcast %4 : tensor<1x256xi32> -> tensor<512x256xi32>
    %mkoff = arith.addi %moff, %koff : tensor<512x256xi32>
    // afloat pointer
    %8 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<512x256x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %mkoff : tensor<512x256x!tt.ptr<bf16>>, tensor<512x256xi32>
    // res pointer
    %18 = tt.splat %res : !tt.ptr<bf16> -> tensor<512x!tt.ptr<bf16>>
    %19 = tt.addptr %18, %0 : tensor<512x!tt.ptr<bf16>>, tensor<512xi32>
    %afm = tt.load %9 : tensor<512x256x!tt.ptr<bf16>>
    %5 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 1 : i32} : (tensor<512x256xbf16>) -> tensor<512xbf16>
    tt.store %19, %5 : tensor<512x!tt.ptr<bf16>>
    tt.return
    }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:          %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, 
// CHECK-SAME:          %[[VAL_0:.*]]: memref<?xbf16> {tt.tensor_kind = 0 : i32}, %[[VAL_1:.*]]: memref<?xbf16> {tt.tensor_kind = 1 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32)
// CHECK-SAME:          attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK:           %[[VAL_7:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: [0], sizes: [512, 256], strides: [256, 1] : memref<?xbf16> to memref<512x256xbf16, strided<[256, 1]>>
// CHECK:           %[[VAL_8:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: [0], sizes: [512], strides: [1] : memref<?xbf16> to memref<512xbf16, strided<[1]>>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<512x256xbf16>
// CHECK:           memref.copy %[[VAL_7]], %[[VAL_9]] : memref<512x256xbf16, strided<[256, 1]>> to memref<512x256xbf16>
// CHECK:           %[[VAL_10:.*]] = bufferization.to_tensor %[[VAL_9]] restrict writable : memref<512x256xbf16>
// CHECK:           %[[VAL_11:.*]] = tensor.empty() : tensor<512xbf16>
// CHECK:           %[[VAL_14:.*]] = linalg.fill ins(%[[VAL_6]] : bf16) outs(%[[VAL_11]] : tensor<512xbf16>) -> tensor<512xbf16>
// CHECK:           %[[VAL_15:.*]] = linalg.reduce ins(%[[VAL_10]] : tensor<512x256xbf16>) outs(%[[VAL_14]] : tensor<512xbf16>) dimensions = [1]
// CHECK:             (%[[VAL_16:.*]]: bf16, %[[VAL_17:.*]]: bf16) {
// CHECK:               %[[VAL_18:.*]] = arith.addf %[[VAL_16]], %[[VAL_17]] : bf16
// CHECK:               linalg.yield %[[VAL_18]] : bf16
// CHECK:             }
// CHECK:           bufferization.materialize_in_destination %[[VAL_15]] in writable %[[VAL_8]] : (tensor<512xbf16>, memref<512xbf16, strided<[1]>>) -> ()
// CHECK:           return
// CHECK:         }
