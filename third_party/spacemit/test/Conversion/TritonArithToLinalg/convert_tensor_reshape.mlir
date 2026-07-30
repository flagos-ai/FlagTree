// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func public @bcast_kernel_01(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c32_i32 = arith.constant 32 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32_i32 : i32
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : i32 -> tensor<32xi32>
    %4 = arith.addi %3, %2 : tensor<32xi32>
    %5 = tt.make_range {end = 2048 : i32, start = 0 : i32} : tensor<2048xi32>
    %6 = tt.splat %1 : i32 -> tensor<2048xi32>
    %7 = arith.addi %6, %5 : tensor<2048xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %9 = tt.addptr %8, %4 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %10 = tt.load %9 : tensor<32x!tt.ptr<f32>>
    %11 = tt.reshape %10 allow_reorder : tensor<32xf32> -> tensor<1x32xf32>
    %12 = tt.broadcast %11 : tensor<1x32xf32> -> tensor<64x32xf32>
    %13 = tt.reshape %12 allow_reorder : tensor<64x32xf32> -> tensor<2048xf32>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2048x!tt.ptr<f32>>
    %15 = tt.addptr %14, %7 : tensor<2048x!tt.ptr<f32>>, tensor<2048xi32>
    tt.store %15, %13 : tensor<2048x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @bcast_kernel_01
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[PARAM_5_]], [[CST_32_]] : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = tensor.empty() : tensor<32xi32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_1_]] : tensor<32xi32>) {
// CHECK:           ^bb0([[out_:%.*]]: i32):
// CHECK:             [[VAR_22_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_23_:%.+]] = arith.index_cast [[VAR_22_]] : index to i32
// CHECK:             linalg.yield [[VAR_23_]] : i32
// CHECK:           } -> tensor<32xi32>
// CHECK:           [[VAR_3_:%.+]] = tensor.empty() : tensor<32xi32>
// CHECK:           [[VAR_4_:%.+]] = linalg.fill ins([[VAR_0_]] : i32) outs([[VAR_3_]] : tensor<32xi32>) -> tensor<32xi32>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_4_]], [[VAR_2_]] : tensor<32xi32>, tensor<32xi32>) outs([[VAR_4_]] : tensor<32xi32>) {
// CHECK:           ^bb0([[in_:%.*]]: i32, [[in_1_:%.*]]: i32, [[out_]]: i32):
// CHECK:             [[VAR_22_1_:%.+]] = arith.addi [[in_]], [[in_1_]] : i32
// CHECK:             linalg.yield [[VAR_22_1_]] : i32
// CHECK:           } -> tensor<32xi32>
// CHECK:           [[VAR_6_:%.+]] = tensor.empty() : tensor<2048xi32>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_6_]] : tensor<2048xi32>) {
// CHECK:           ^bb0([[out_]]: i32):
// CHECK:             [[VAR_22_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_23_1_:%.+]] = arith.index_cast [[VAR_22_2_]] : index to i32
// CHECK:             linalg.yield [[VAR_23_1_]] : i32
// CHECK:           } -> tensor<2048xi32>
// CHECK:           [[VAR_8_:%.+]] = tensor.empty() : tensor<2048xi32>
// CHECK:           [[VAR_9_:%.+]] = linalg.fill ins([[VAR_0_]] : i32) outs([[VAR_8_]] : tensor<2048xi32>) -> tensor<2048xi32>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_9_]], [[VAR_7_]] : tensor<2048xi32>, tensor<2048xi32>) outs([[VAR_9_]] : tensor<2048xi32>) {
// CHECK:           ^bb0([[in_]]: i32, [[in_1_]]: i32, [[out_]]: i32):
// CHECK:             [[VAR_22_3_:%.+]] = arith.addi [[in_]], [[in_1_]] : i32
// CHECK:             linalg.yield [[VAR_22_3_]] : i32
// CHECK:           } -> tensor<2048xi32>
// CHECK:           [[VAR_11_:%.+]] = tensor.empty() : tensor<32x!tt.ptr<f32>>
// CHECK:           [[VAR_12_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<f32>) outs([[VAR_11_]] : tensor<32x!tt.ptr<f32>>) -> tensor<32x!tt.ptr<f32>>
// CHECK:           [[VAR_13_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_12_]], [[VAR_5_]] : tensor<32x!tt.ptr<f32>>, tensor<32xi32>) outs([[VAR_12_]] : tensor<32x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_1_]]: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_22_4_:%.+]] = tt.addptr [[in_]], [[in_1_]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_22_4_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<32x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_13_MEM_:%.+]] = tt.load [[VAR_13_]] : tensor<32x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_expanded_:%.+]] = tensor.expand_shape [[LOAD_VAR_13_MEM_]] {{.}}[0, 1]{{.}} output_shape [1, 32] : tensor<32xf32> into tensor<1x32xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = tensor.empty() : tensor<64x32xf32>
// CHECK:           [[VAR_16_:%.+]] = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_]] : tensor<1x32xf32>) outs([[VAR_15_]] : tensor<64x32xf32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_]]: f32, [[out_]]: f32):
// CHECK:             linalg.yield [[in_]] : f32
// CHECK:           } -> tensor<64x32xf32>
// CHECK-DAG:       [[VAR_collapsed_:%.+]] = tensor.collapse_shape [[VAR_16_]] {{.}}[0, 1]{{.}} : tensor<64x32xf32> into tensor<2048xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = tensor.empty() : tensor<2048x!tt.ptr<f32>>
// CHECK:           [[VAR_20_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<f32>) outs([[VAR_19_]] : tensor<2048x!tt.ptr<f32>>) -> tensor<2048x!tt.ptr<f32>>
// CHECK:           [[VAR_21_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_20_]], [[VAR_10_]] : tensor<2048x!tt.ptr<f32>>, tensor<2048xi32>) outs([[VAR_20_]] : tensor<2048x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_1_]]: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_22_5_:%.+]] = tt.addptr [[in_]], [[in_1_]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_22_5_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<2048x!tt.ptr<f32>>
// CHECK:           tt.store [[VAR_21_]], [[VAR_collapsed_]] : tensor<2048x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
