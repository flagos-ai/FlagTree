// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %a : !tt.ptr<f32>,
    %b : !tt.ptr<f32>,
    %c : tensor<1024x!tt.ptr<f32>>
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
        tt.store %c, %6 : tensor<1024x!tt.ptr<f32>>
        tt.return
    }
}
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: tensor<1024x!tt.ptr<f32>>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<1024xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<1024xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_17_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_18_:%.+]] = arith.index_cast [[VAR_17_]] : index to i32
// CHECK:             linalg.yield [[VAR_18_]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           [[VAR_2_:%.+]] = tensor.empty() : tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<f32>) outs([[VAR_2_]] : tensor<1024x!tt.ptr<f32>>) -> tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_3_]], [[VAR_1_]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs([[VAR_3_]] : tensor<1024x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_:%.+]]: !tt.ptr<f32>, [[in_]]_0: i32, [[out_:%.+]]: !tt.ptr<f32>):
// CHECK:             [[VAR_17_1_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_17_1_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_5_:%.+]] = tensor.empty() : tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_6_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<f32>) outs([[VAR_5_]] : tensor<1024x!tt.ptr<f32>>) -> tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_6_]], [[VAR_1_]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs([[VAR_6_]] : tensor<1024x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_0: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_17_2_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_17_2_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<1024x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_4_MEM_:%.+]] = tt.load [[VAR_4_]] : tensor<1024x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_7_MEM_:%.+]] = tt.load [[VAR_7_]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[LOAD_VAR_4_MEM_]], [[LOAD_VAR_7_MEM_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[LOAD_VAR_4_MEM_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_]]: f32, [[in_]]_0: f32, [[out_]]: f32):
// CHECK:             [[VAR_17_3_:%.+]] = arith.addf [[in_]], [[in_]]_0 : f32
// CHECK:             linalg.yield [[VAR_17_3_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_11_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_10_]], [[LOAD_VAR_7_MEM_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_10_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_]]: f32, [[in_]]_0: f32, [[out_]]: f32):
// CHECK:             [[VAR_17_4_:%.+]] = arith.subf [[in_]], [[in_]]_0 : f32
// CHECK:             linalg.yield [[VAR_17_4_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_12_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_11_]], [[LOAD_VAR_7_MEM_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_11_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_]]: f32, [[in_]]_0: f32, [[out_]]: f32):
// CHECK:             [[VAR_17_5_:%.+]] = arith.mulf [[in_]], [[in_]]_0 : f32
// CHECK:             linalg.yield [[VAR_17_5_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_13_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_12_]], [[LOAD_VAR_7_MEM_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_12_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_]]: f32, [[in_]]_0: f32, [[out_]]: f32):
// CHECK:             [[VAR_17_6_:%.+]] = arith.divf [[in_]], [[in_]]_0 : f32
// CHECK:             linalg.yield [[VAR_17_6_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_14_:%.+]] = tensor.empty() : tensor<1024xi1>
// CHECK:           [[VAR_15_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_13_]], [[LOAD_VAR_7_MEM_]] : tensor<1024xf32>, tensor<1024xf32>) outs([[VAR_14_]] : tensor<1024xi1>) {
// CHECK:           ^bb0([[in_]]: f32, [[in_]]_0: f32, [[out_]]: i1):
// CHECK:             [[VAR_17_7_:%.+]] = arith.cmpf oeq, [[in_]], [[in_]]_0 : f32
// CHECK:             linalg.yield [[VAR_17_7_]] : i1
// CHECK:           } -> tensor<1024xi1>
// CHECK:           [[VAR_16_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_15_]], [[LOAD_VAR_4_MEM_]], [[LOAD_VAR_7_MEM_]] : tensor<1024xi1>, tensor<1024xf32>, tensor<1024xf32>) outs([[LOAD_VAR_4_MEM_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_]]: i1, [[in_]]_0: f32, [[in_]]_1: f32, [[out_]]: f32):
// CHECK:             [[VAR_17_8_:%.+]] = arith.select [[in_]], [[in_]]_0, [[in_]]_1 : f32
// CHECK:             linalg.yield [[VAR_17_8_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           tt.store [[PARAM_2_]], [[VAR_16_]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
