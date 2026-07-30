// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
    tt.func @kernel(%fin : f32,
                    %bin : bf16,
                    %save_ptr0 : !tt.ptr<f32>,
                    %save_ptr1 : !tt.ptr<bf16>) -> () {
        %0 = tt.splat %fin : f32 -> tensor<1024xf32>
        %1 = tt.splat %bin : bf16 -> tensor<128x256xbf16>
        // save pointers, intentionally splat the base pointer for brevity
        %save0 = tt.splat %save_ptr0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %save1 = tt.splat %save_ptr1 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
        tt.store %save0, %0 : tensor<1024x!tt.ptr<f32>>
        tt.store %save1, %1 : tensor<128x256x!tt.ptr<bf16>>
        tt.return
    }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @kernel
// CHECK-SAME:    ([[PARAM_0_:%.+]]: f32, [[PARAM_1_:%.+]]: bf16, [[PARAM_2_:%.+]]: memref<*xf32>, [[PARAM_3_:%.+]]: memref<*xbf16>, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_empty_offsets_1d_:%.+]] = tensor.empty() : tensor<1024xi32>
// CHECK-DAG:       [[VAR_zero_offsets_1d_:%.+]] = linalg.fill ins([[CST_0_]] : i32) outs([[VAR_empty_offsets_1d_]] : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK-DAG:       [[VAR_empty_offsets_2d_:%.+]] = tensor.empty() : tensor<128x256xi32>
// CHECK-DAG:       [[VAR_zero_offsets_2d_:%.+]] = linalg.fill ins([[CST_0_]] : i32) outs([[VAR_empty_offsets_2d_]] : tensor<128x256xi32>) -> tensor<128x256xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[PARAM_0_]] : f32) outs([[VAR_0_]] : tensor<1024xf32>) -> tensor<1024xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tensor.empty() : tensor<128x256xbf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_1_]] : bf16) outs([[VAR_2_]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
// CHECK:           [[VAR_cast_2_:%.+]] = memref.cast [[PARAM_2_]] : memref<*xf32> to memref<?xf32>
// CHECK:           linalg.generic {indexing_maps = [[[MAP_0_]], [[MAP_0_]]], iterator_types = ["parallel"]} ins([[VAR_zero_offsets_1d_]], [[VAR_1_]] : tensor<1024xi32>, tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[IN_0_]] : i32 to index
// CHECK:             memref.store [[IN_1_]], [[VAR_cast_2_]]{{.}}[[VAR_4_]]{{.}} : memref<?xf32>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           [[VAR_cast_3_:%.+]] = memref.cast [[PARAM_3_]] : memref<*xbf16> to memref<?xbf16>
// CHECK:           linalg.generic {indexing_maps = [[[MAP_1_]], [[MAP_1_]]], iterator_types = ["parallel", "parallel"]} ins([[VAR_zero_offsets_2d_]], [[VAR_3_]] : tensor<128x256xi32>, tensor<128x256xbf16>) {
// CHECK:           ^bb0([[IN_2_:%.+]]: i32, [[IN_3_:%.+]]: bf16):
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[IN_2_]] : i32 to index
// CHECK:             memref.store [[IN_3_]], [[VAR_cast_3_]]{{.}}[[VAR_5_]]{{.}} : memref<?xbf16>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
