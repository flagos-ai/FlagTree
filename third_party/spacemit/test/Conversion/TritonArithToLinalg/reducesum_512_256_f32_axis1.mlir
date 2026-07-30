// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<f32>,
        %res : !tt.ptr<f32>
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
    %8 = tt.splat %afloat : !tt.ptr<f32> -> tensor<512x256x!tt.ptr<f32>>
    %9 = tt.addptr %8, %mkoff : tensor<512x256x!tt.ptr<f32>>, tensor<512x256xi32>
    // res pointer
    %18 = tt.splat %res : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    %afm = tt.load %9 : tensor<512x256x!tt.ptr<f32>>
    %5 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.addf %arg5, %arg6 : f32
      tt.reduce.return %21 : f32
    }) {axis = 1 : i32} : (tensor<512x256xf32>) -> tensor<512xf32>
    tt.store %19, %5 : tensor<512x!tt.ptr<f32>>
    tt.return
    }
}
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<512xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_256_]] : i32) outs([[VAR_0_]] : tensor<512xi32>) -> tensor<512xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tensor.empty() : tensor<512xi32>
// CHECK:           [[VAR_3_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_2_]] : tensor<512xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_22_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_23_:%.+]] = arith.index_cast [[VAR_22_]] : index to i32
// CHECK:             linalg.yield [[VAR_23_]] : i32
// CHECK:           } -> tensor<512xi32>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_3_]], [[VAR_1_]] : tensor<512xi32>, tensor<512xi32>) outs([[VAR_3_]] : tensor<512xi32>) {
// CHECK:           ^bb0([[in_:%.+]]: i32, [[in_]]_1: i32, [[out_]]: i32):
// CHECK:             [[VAR_22_1_:%.+]] = arith.muli [[in_]], [[in_]]_1 : i32
// CHECK:             linalg.yield [[VAR_22_1_]] : i32
// CHECK:           } -> tensor<512xi32>
// CHECK-DAG:       [[VAR_expanded_:%.+]] = tensor.expand_shape [[VAR_4_]] {{.}}[0, 1]{{.}} output_shape [512, 1] : tensor<512xi32> into tensor<512x1xi32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tensor.empty() : tensor<512x256xi32>
// CHECK:           [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_]] : tensor<512x1xi32>) outs([[VAR_5_]] : tensor<512x256xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<512x256xi32>
// CHECK:           [[VAR_7_:%.+]] = tensor.empty() : tensor<256xi32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_7_]] : tensor<256xi32>) {
// CHECK:           ^bb0([[out_]]: i32):
// CHECK:             [[VAR_22_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_23_1_:%.+]] = arith.index_cast [[VAR_22_2_]] : index to i32
// CHECK:             linalg.yield [[VAR_23_1_]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK-DAG:       [[VAR_expanded_0_:%.+]] = tensor.expand_shape [[VAR_8_]] {{.}}[0, 1]{{.}} output_shape [1, 256] : tensor<256xi32> into tensor<1x256xi32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tensor.empty() : tensor<512x256xi32>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_0_]] : tensor<1x256xi32>) outs([[VAR_9_]] : tensor<512x256xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<512x256xi32>
// CHECK:           [[VAR_11_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_6_]], [[VAR_10_]] : tensor<512x256xi32>, tensor<512x256xi32>) outs([[VAR_6_]] : tensor<512x256xi32>) {
// CHECK:           ^bb0([[in_]]: i32, [[in_]]_1: i32, [[out_]]: i32):
// CHECK:             [[VAR_22_3_:%.+]] = arith.addi [[in_]], [[in_]]_1 : i32
// CHECK:             linalg.yield [[VAR_22_3_]] : i32
// CHECK:           } -> tensor<512x256xi32>
// CHECK:           [[VAR_12_:%.+]] = tensor.empty() : tensor<512x256x!tt.ptr<f32>>
// CHECK:           [[VAR_13_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<f32>) outs([[VAR_12_]] : tensor<512x256x!tt.ptr<f32>>) -> tensor<512x256x!tt.ptr<f32>>
// CHECK:           [[VAR_14_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_13_]], [[VAR_11_]] : tensor<512x256x!tt.ptr<f32>>, tensor<512x256xi32>) outs([[VAR_13_]] : tensor<512x256x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_1: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_22_4_:%.+]] = tt.addptr [[in_]], [[in_]]_1 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_22_4_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<512x256x!tt.ptr<f32>>
// CHECK:           [[VAR_15_:%.+]] = tensor.empty() : tensor<512x!tt.ptr<f32>>
// CHECK:           [[VAR_16_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<f32>) outs([[VAR_15_]] : tensor<512x!tt.ptr<f32>>) -> tensor<512x!tt.ptr<f32>>
// CHECK:           [[VAR_17_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_16_]], [[VAR_3_]] : tensor<512x!tt.ptr<f32>>, tensor<512xi32>) outs([[VAR_16_]] : tensor<512x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_1: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_22_5_:%.+]] = tt.addptr [[in_]], [[in_]]_1 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_22_5_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<512x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_14_MEM_:%.+]] = tt.load [[VAR_14_]] : tensor<512x256x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_19_:%.+]] = tensor.empty() : tensor<256x512xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_transposed_:%.+]] = linalg.transpose ins([[LOAD_VAR_14_MEM_]] : tensor<512x256xf32>) outs([[VAR_19_]] : tensor<256x512xf32>) permutation = [1, 0]
// CHECK-DAG:       [[VAR_20_:%.+]] = tensor.empty() : tensor<512xf32>
// CHECK:           [[VAR_21_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_20_]] : tensor<512xf32>) -> tensor<512xf32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_transposed_]] : tensor<256x512xf32>) outs([[VAR_21_]] : tensor<512xf32>) dimensions = [0]
// CHECK:             ([[in_]]: f32, [[in_]]it: f32) {
// CHECK:               [[VAR_22_6_:%.+]] = arith.addf [[in_]], [[in_]]it : f32
// CHECK:               linalg.yield [[VAR_22_6_]] : f32
// CHECK:             }
// CHECK:           tt.store [[VAR_17_]], [[VAR_reduced_]] : tensor<512x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
