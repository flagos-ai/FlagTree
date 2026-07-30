// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
    %a : !tt.ptr<f32>,
    %b : !tt.ptr<f32>,
    %c : !tt.ptr<f32>
  ) -> () {
        %cst = arith.constant dense<true> : tensor<1024xi1>
        // offset calculations
        %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
        // a pointer
        %8 = tt.splat %a : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        // b pointer
        %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
        %am = tt.load %9 : tensor<1024x!tt.ptr<f32>>
        %bm = tt.load %19 : tensor<1024x!tt.ptr<f32>>
        %1 = arith.addf %am, %bm : tensor<1024xf32>
        %2 = arith.subf %1, %bm : tensor<1024xf32>
        %3 = arith.mulf %2, %bm : tensor<1024xf32>
        %4 = arith.divf %3, %bm : tensor<1024xf32>
        %5 = arith.cmpf "oeq", %4, %bm : tensor<1024xf32>
        %6 = arith.select %5, %am, %bm : tensor<1024xi1>, tensor<1024xf32>
        // c pointer, intentionally splat the base pointer for brevity
        %c_out = tt.splat %c : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
        tt.store %c_out, %6 : tensor<1024x!tt.ptr<f32>>
        tt.return
    }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: memref<*xf32>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_empty_offsets_:%.+]] = tensor.empty() : tensor<1024xi32>
// CHECK-DAG:       [[VAR_zero_offsets_:%.+]] = linalg.fill ins([[CST_0_]] : i32) outs([[VAR_empty_offsets_]] : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<1024xf32, strided<[1]>> to memref<1024xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1024xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_0_]], [[RES_1_]] : memref<1024xf32, strided<[1]>> to memref<1024xf32>
// CHECK:           [[VAR_1_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<1024xf32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_0_]], [[VAR_1_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_0_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32, [[IN_2_:%.+]]: f32):
// CHECK:             [[VAR_9_:%.+]] = arith.addf [[IN_0_]], [[IN_1_]] : f32
// CHECK:             linalg.yield [[VAR_9_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_3_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_2_]], [[VAR_1_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_2_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_3_:%.+]]: f32, [[IN_4_:%.+]]: f32, [[IN_5_:%.+]]: f32):
// CHECK:             [[VAR_9_1_:%.+]] = arith.subf [[IN_3_]], [[IN_4_]] : f32
// CHECK:             linalg.yield [[VAR_9_1_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_3_]], [[VAR_1_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_3_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_6_:%.+]]: f32, [[IN_7_:%.+]]: f32, [[IN_8_:%.+]]: f32):
// CHECK:             [[VAR_9_2_:%.+]] = arith.mulf [[IN_6_]], [[IN_7_]] : f32
// CHECK:             linalg.yield [[VAR_9_2_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_4_]], [[VAR_1_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_4_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_9_:%.+]]: f32, [[IN_10_:%.+]]: f32, [[IN_11_:%.+]]: f32):
// CHECK:             [[VAR_9_3_:%.+]] = arith.divf [[IN_9_]], [[IN_10_]] : f32
// CHECK:             linalg.yield [[VAR_9_3_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_6_:%.+]] = tensor.empty() : tensor<1024xi1>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_5_]], [[VAR_1_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_6_]] : tensor<1024xi1>) {
// CHECK:           ^bb0([[IN_12_:%.+]]: f32, [[IN_13_:%.+]]: f32, [[IN_14_:%.+]]: i1):
// CHECK:             [[VAR_9_4_:%.+]] = arith.cmpf oeq, [[IN_12_]], [[IN_13_]] : f32
// CHECK:             linalg.yield [[VAR_9_4_]] : i1
// CHECK:           } -> tensor<1024xi1>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]], [[VAR_0_]], [[VAR_1_]] : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_0_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_15_:%.+]]: i1, [[IN_16_:%.+]]: f32, [[IN_17_:%.+]]: f32, [[IN_18_:%.+]]: f32):
// CHECK:             [[VAR_9_5_:%.+]] = arith.select [[IN_15_]], [[IN_16_]], [[IN_17_]] : f32
// CHECK:             linalg.yield [[VAR_9_5_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_cast_:%.+]] = memref.cast [[PARAM_2_]] : memref<*xf32> to memref<?xf32>
// CHECK:           linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_zero_offsets_]], [[VAR_8_]] : tensor<1024xi32>, tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_18_:%.+]]: i32, [[IN_19_:%.+]]: f32):
// CHECK:             [[VAR_10_:%.+]] = arith.index_cast [[IN_18_]] : i32 to index
// CHECK:             memref.store [[IN_19_]], [[VAR_cast_]]{{.}}[[VAR_10_]]{{.}} : memref<?xf32>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
