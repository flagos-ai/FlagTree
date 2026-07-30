// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
    %a : !tt.ptr<i1>,
    %b : !tt.ptr<f32>,
    %c : !tt.ptr<f32>,
    %d : !tt.ptr<f32>
  ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // a pointer
    %8 = tt.splat %a : !tt.ptr<i1> -> tensor<1024x!tt.ptr<i1>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i1>>, tensor<1024xi32>
    // b pointer
    %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // c pointer
    %28 = tt.splat %c : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %29 = tt.addptr %28, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %am = tt.load %9 : tensor<1024x!tt.ptr<i1>>
    %bm = tt.load %19 : tensor<1024x!tt.ptr<f32>>
    %cm = tt.load %29 : tensor<1024x!tt.ptr<f32>>
    %10 = arith.select %am, %bm, %cm : tensor<1024xi1>, tensor<1024xf32>
    // d pointer, intentionally splat the base pointer for brevity
    %d_out = tt.splat %d : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    tt.store %d_out, %10 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi1>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: memref<*xf32>, [[PARAM_3_:%.+]]: memref<*xf32>, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_empty_offsets_:%.+]] = tensor.empty() : tensor<1024xi32>
// CHECK-DAG:       [[VAR_zero_offsets_:%.+]] = linalg.fill ins([[CST_0_]] : i32) outs([[VAR_empty_offsets_]] : tensor<1024xi32>) -> tensor<1024xi32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1024], strides: [1] : memref<*xi1> to memref<1024xi1, strided<[1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK-DAG:       [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: [0], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1024xi1>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<1024xi1, strided<[1]>> to memref<1024xi1>
// CHECK-DAG:       [[VAR_0_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1024xi1>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_0_]], [[RES_1_]] : memref<1024xf32, strided<[1]>> to memref<1024xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<1024xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_1_]], [[RES_2_]] : memref<1024xf32, strided<[1]>> to memref<1024xf32>
// CHECK:           [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_2_]] restrict writable : memref<1024xf32>
// CHECK:           [[VAR_3_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_0_]], [[VAR_1_]], [[VAR_2_]] : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_1_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i1, [[IN_1_:%.+]]: f32, [[IN_2_:%.+]]: f32, [[IN_3_:%.+]]: f32):
// CHECK:             [[VAR_4_:%.+]] = arith.select [[IN_0_]], [[IN_1_]], [[IN_2_]] : f32
// CHECK:             linalg.yield [[VAR_4_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_cast_:%.+]] = memref.cast [[PARAM_3_]] : memref<*xf32> to memref<?xf32>
// CHECK:           linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_zero_offsets_]], [[VAR_3_]] : tensor<1024xi32>, tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_3_:%.+]]: i32, [[IN_4_:%.+]]: f32):
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[IN_3_]] : i32 to index
// CHECK:             memref.store [[IN_4_]], [[VAR_cast_]]{{.}}[[VAR_5_]]{{.}} : memref<?xf32>
// CHECK:             linalg.yield
// CHECK:           }
// CHECK:           return
// CHECK:         }
