// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %a : !tt.ptr<i1>,
    %b : !tt.ptr<f32>,
    %c : !tt.ptr<f32>,
    %d : tensor<1024x!tt.ptr<f32>>
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
    tt.store %d, %10 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<i1>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: !tt.ptr<f32>, [[PARAM_3_:%.+]]: tensor<1024x!tt.ptr<f32>>, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<1024xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<1024xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_15_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_16_:%.+]] = arith.index_cast [[VAR_15_]] : index to i32
// CHECK:             linalg.yield [[VAR_16_]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           [[VAR_2_:%.+]] = tensor.empty() : tensor<1024x!tt.ptr<i1>>
// CHECK:           [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<i1>) outs([[VAR_2_]] : tensor<1024x!tt.ptr<i1>>) -> tensor<1024x!tt.ptr<i1>>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_3_]], [[VAR_1_]] : tensor<1024x!tt.ptr<i1>>, tensor<1024xi32>) outs([[VAR_3_]] : tensor<1024x!tt.ptr<i1>>) {
// CHECK:           ^bb0([[in_:%.+]]: !tt.ptr<i1>, [[in_]]_0: i32, [[out_:%.+]]: !tt.ptr<i1>):
// CHECK:             [[VAR_15_1_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<i1>, i32
// CHECK:             linalg.yield [[VAR_15_1_]] : !tt.ptr<i1>
// CHECK:           } -> tensor<1024x!tt.ptr<i1>>
// CHECK:           [[VAR_5_:%.+]] = tensor.empty() : tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_6_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<f32>) outs([[VAR_5_]] : tensor<1024x!tt.ptr<f32>>) -> tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_6_]], [[VAR_1_]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs([[VAR_6_]] : tensor<1024x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_0: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_15_2_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_15_2_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_8_:%.+]] = tensor.empty() : tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_9_:%.+]] = linalg.fill ins([[PARAM_2_]] : !tt.ptr<f32>) outs([[VAR_8_]] : tensor<1024x!tt.ptr<f32>>) -> tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_9_]], [[VAR_1_]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs([[VAR_9_]] : tensor<1024x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_0: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_15_3_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_15_3_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<1024x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_4_MEM_:%.+]] = tt.load [[VAR_4_]] : tensor<1024x!tt.ptr<i1>>
// CHECK-DAG:       [[LOAD_VAR_7_MEM_:%.+]] = tt.load [[VAR_7_]] : tensor<1024x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_10_MEM_:%.+]] = tt.load [[VAR_10_]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_14_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins([[LOAD_VAR_4_MEM_]], [[LOAD_VAR_7_MEM_]], [[LOAD_VAR_10_MEM_]] : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs([[LOAD_VAR_7_MEM_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_]]: i1, [[in_]]_0: f32, [[in_]]_1: f32, [[out_]]: f32):
// CHECK:             [[VAR_15_4_:%.+]] = arith.select [[in_]], [[in_]]_0, [[in_]]_1 : f32
// CHECK:             linalg.yield [[VAR_15_4_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           tt.store [[PARAM_3_]], [[VAR_14_]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
