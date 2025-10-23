// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(%fin : f32,
                    %bin : bf16,
                    %save0 : tensor<1024x!tt.ptr<f32>>,
                    %save1 : tensor<128x256x!tt.ptr<bf16>>) -> () {
        %0 = tt.splat %fin : f32 -> tensor<1024xf32>
        %1 = tt.splat %bin : bf16 -> tensor<128x256xbf16>
        tt.store %save0, %0 : tensor<1024x!tt.ptr<f32>>
        tt.store %save1, %1 : tensor<128x256x!tt.ptr<bf16>>
        tt.return
    }
}
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:      %[[ARG_0:.*]]: memref<?xi8>, %[[ARG_1:.*]]: memref<?xi8>, %[[VAL_0:.*]]: f32, %[[VAL_1:.*]]: bf16, %[[VAL_2:.*]]: memref<1024xf32>, %[[VAL_3:.*]]: memref<128x256xbf16>, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<1024xf32>
// CHECK:           %[[VAL_8:.*]] = linalg.fill ins(%[[VAL_0]] : f32) outs(%[[VAL_7]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK:           %[[VAL_9:.*]] = tensor.empty() : tensor<128x256xbf16>
// CHECK:           %[[VAL_10:.*]] = linalg.fill ins(%[[VAL_1]] : bf16) outs(%[[VAL_9]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
// CHECK:           bufferization.materialize_in_destination %[[VAL_8]] in writable %[[VAL_2]]
// CHECK:           bufferization.materialize_in_destination %[[VAL_10]] in writable %[[VAL_3]]
// CHECK:           return
// CHECK:         }
