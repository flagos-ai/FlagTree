// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %a : !tt.ptr<f32>,
    %b : !tt.ptr<f32>,
    %c : tensor<128x128x!tt.ptr<f32>>,
    %d : tensor<128x128x!tt.ptr<f32>>
  ) -> () {
        // offset calculations
        %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
        %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
        %moff = tt.broadcast %1 : tensor<128x1xi32> -> tensor<128x128xi32>
        %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
        %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
        %koff = tt.broadcast %4 : tensor<1x128xi32> -> tensor<128x128xi32>
        %mkoff = arith.addi %moff, %koff : tensor<128x128xi32>
        // a pointer
        %8 = tt.splat %a : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
        %9 = tt.addptr %8, %mkoff : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
        // b pointer
        %18 = tt.splat %b : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
        %19 = tt.addptr %18, %mkoff : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
        %af = tt.load %9 : tensor<128x128x!tt.ptr<f32>>
        %bf = tt.load %19 : tensor<128x128x!tt.ptr<f32>>
        %res0 = arith.addf %af, %bf : tensor<128x128xf32>
        %res1 = arith.subf %af, %bf : tensor<128x128xf32>
        tt.store %c, %res0 : tensor<128x128x!tt.ptr<f32>>
        tt.store %d, %res1 : tensor<128x128x!tt.ptr<f32>>
        tt.return
    }
}
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: tensor<128x128x!tt.ptr<f32>>, [[PARAM_3_:%.+]]: tensor<128x128x!tt.ptr<f32>>, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<128xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<128xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_19_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_20_:%.+]] = arith.index_cast [[VAR_19_]] : index to i32
// CHECK:             linalg.yield [[VAR_20_]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK-DAG:       [[VAR_expanded_:%.+]] = tensor.expand_shape [[VAR_1_]] {{.}}[0, 1]{{.}} output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tensor.empty() : tensor<128x128xi32>
// CHECK:           [[VAR_3_:%.+]] = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_]] : tensor<128x1xi32>) outs([[VAR_2_]] : tensor<128x128xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0([[in_:%.+]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<128x128xi32>
// CHECK:           [[VAR_4_:%.+]] = tensor.empty() : tensor<128xi32>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_4_]] : tensor<128xi32>) {
// CHECK:           ^bb0([[out_]]: i32):
// CHECK:             [[VAR_19_1_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_20_1_:%.+]] = arith.index_cast [[VAR_19_1_]] : index to i32
// CHECK:             linalg.yield [[VAR_20_1_]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK-DAG:       [[VAR_expanded_0_:%.+]] = tensor.expand_shape [[VAR_5_]] {{.}}[0, 1]{{.}} output_shape [1, 128] : tensor<128xi32> into tensor<1x128xi32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tensor.empty() : tensor<128x128xi32>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_0_]] : tensor<1x128xi32>) outs([[VAR_6_]] : tensor<128x128xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<128x128xi32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_3_]], [[VAR_7_]] : tensor<128x128xi32>, tensor<128x128xi32>) outs([[VAR_3_]] : tensor<128x128xi32>) {
// CHECK:           ^bb0([[in_]]: i32, [[in_]]_1: i32, [[out_]]: i32):
// CHECK:             [[VAR_19_2_:%.+]] = arith.addi [[in_]], [[in_]]_1 : i32
// CHECK:             linalg.yield [[VAR_19_2_]] : i32
// CHECK:           } -> tensor<128x128xi32>
// CHECK:           [[VAR_9_:%.+]] = tensor.empty() : tensor<128x128x!tt.ptr<f32>>
// CHECK:           [[VAR_10_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<f32>) outs([[VAR_9_]] : tensor<128x128x!tt.ptr<f32>>) -> tensor<128x128x!tt.ptr<f32>>
// CHECK:           [[VAR_11_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_10_]], [[VAR_8_]] : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>) outs([[VAR_10_]] : tensor<128x128x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_1: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_19_3_:%.+]] = tt.addptr [[in_]], [[in_]]_1 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_19_3_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<128x128x!tt.ptr<f32>>
// CHECK:           [[VAR_12_:%.+]] = tensor.empty() : tensor<128x128x!tt.ptr<f32>>
// CHECK:           [[VAR_13_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<f32>) outs([[VAR_12_]] : tensor<128x128x!tt.ptr<f32>>) -> tensor<128x128x!tt.ptr<f32>>
// CHECK:           [[VAR_14_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_13_]], [[VAR_8_]] : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>) outs([[VAR_13_]] : tensor<128x128x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_1: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_19_4_:%.+]] = tt.addptr [[in_]], [[in_]]_1 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_19_4_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<128x128x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_11_MEM_:%.+]] = tt.load [[VAR_11_]] : tensor<128x128x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_14_MEM_:%.+]] = tt.load [[VAR_14_]] : tensor<128x128x!tt.ptr<f32>>
// CHECK:           [[VAR_17_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[LOAD_VAR_11_MEM_]], [[LOAD_VAR_14_MEM_]] : tensor<128x128xf32>, tensor<128x128xf32>) outs([[LOAD_VAR_11_MEM_]] : tensor<128x128xf32>) {
// CHECK:           ^bb0([[in_]]: f32, [[in_]]_1: f32, [[out_]]: f32):
// CHECK:             [[VAR_19_5_:%.+]] = arith.addf [[in_]], [[in_]]_1 : f32
// CHECK:             linalg.yield [[VAR_19_5_]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           [[VAR_18_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[LOAD_VAR_11_MEM_]], [[LOAD_VAR_14_MEM_]] : tensor<128x128xf32>, tensor<128x128xf32>) outs([[LOAD_VAR_11_MEM_]] : tensor<128x128xf32>) {
// CHECK:           ^bb0([[in_]]: f32, [[in_]]_1: f32, [[out_]]: f32):
// CHECK:             [[VAR_19_6_:%.+]] = arith.subf [[in_]], [[in_]]_1 : f32
// CHECK:             linalg.yield [[VAR_19_6_]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           tt.store [[PARAM_2_]], [[VAR_17_]] : tensor<128x128x!tt.ptr<f32>>
// CHECK:           tt.store [[PARAM_3_]], [[VAR_18_]] : tensor<128x128x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
