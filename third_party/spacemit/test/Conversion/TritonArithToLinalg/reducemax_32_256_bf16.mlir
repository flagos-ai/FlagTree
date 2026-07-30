// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
    tt.func @kernel(%afloat : !tt.ptr<bf16>,
        %res : tensor<256x16x!tt.ptr<bf16>>
    ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : i32 -> tensor<32xi32>
    %ws = arith.muli %ct256, %0 : tensor<32xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %m2 = tt.broadcast %1 : tensor<32x1xi32> -> tensor<32x256xi32>
    %100 = tt.expand_dims %m2 {axis = 2 : i32} : tensor<32x256xi32> -> tensor<32x256x1xi32>
    %moff = tt.broadcast %100 : tensor<32x256x1xi32> -> tensor<32x256x16xi32>
    %33 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %34 = tt.expand_dims %33 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %k2 = tt.broadcast %34 : tensor<1x256xi32> -> tensor<32x256xi32>
    %200 = tt.expand_dims %k2 {axis = 2 : i32} : tensor<32x256xi32> -> tensor<32x256x1xi32>
    %koff = tt.broadcast %200 : tensor<32x256x1xi32> -> tensor<32x256x16xi32>
    %23 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %24 = tt.expand_dims %23 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %n2 = tt.broadcast %24 : tensor<1x16xi32> -> tensor<256x16xi32>
    %300 = tt.expand_dims %n2 {axis = 0 : i32} : tensor<256x16xi32> -> tensor<1x256x16xi32>
    %noff = tt.broadcast %300 : tensor<1x256x16xi32> -> tensor<32x256x16xi32>
    %mkoff = arith.addi %moff, %koff : tensor<32x256x16xi32>
    %mknoff = arith.addi %mkoff, %noff : tensor<32x256x16xi32>
    // afloat pointer
    %8 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<32x256x16x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %mknoff : tensor<32x256x16x!tt.ptr<bf16>>, tensor<32x256x16xi32>
    %afm = tt.load %9 : tensor<32x256x16x!tt.ptr<bf16>>
    %6 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.cmpf ogt, %arg5, %arg6 : bf16
      %22 = arith.select %21, %arg5, %arg6 : bf16
      tt.reduce.return %22 : bf16
    }) {axis = 0 : i32} : (tensor<32x256x16xbf16>) -> tensor<256x16xbf16>
    tt.store %res, %6 : tensor<256x16x!tt.ptr<bf16>>
    tt.return
    }
}
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d1, 0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1, d2) -> (0, d1, d2)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: tensor<256x16x!tt.ptr<bf16>>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF80 : bf16
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<32xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_256_]] : i32) outs([[VAR_0_]] : tensor<32xi32>) -> tensor<32xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tensor.empty() : tensor<32xi32>
// CHECK:           [[VAR_3_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_2_]] : tensor<32xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_29_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_30_:%.+]] = arith.index_cast [[VAR_29_]] : index to i32
// CHECK:             linalg.yield [[VAR_30_]] : i32
// CHECK:           } -> tensor<32xi32>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_3_]], [[VAR_1_]] : tensor<32xi32>, tensor<32xi32>) outs([[VAR_3_]] : tensor<32xi32>) {
// CHECK:           ^bb0([[in_:%.+]]: i32, [[in_]]_5: i32, [[out_]]: i32):
// CHECK:             [[VAR_29_1_:%.+]] = arith.muli [[in_]], [[in_]]_5 : i32
// CHECK:             linalg.yield [[VAR_29_1_]] : i32
// CHECK:           } -> tensor<32xi32>
// CHECK-DAG:       [[VAR_expanded_:%.+]] = tensor.expand_shape [[VAR_4_]] {{.}}[0, 1]{{.}} output_shape [32, 1] : tensor<32xi32> into tensor<32x1xi32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tensor.empty() : tensor<32x256xi32>
// CHECK:           [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_]] : tensor<32x1xi32>) outs([[VAR_5_]] : tensor<32x256xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<32x256xi32>
// CHECK-DAG:       [[VAR_expanded_0_:%.+]] = tensor.expand_shape [[VAR_6_]] {{.}}[0], [1, 2]{{.}} output_shape [32, 256, 1] : tensor<32x256xi32> into tensor<32x256x1xi32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tensor.empty() : tensor<32x256x16xi32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins([[VAR_expanded_0_]] : tensor<32x256x1xi32>) outs([[VAR_7_]] : tensor<32x256x16xi32>) attrs =  {broadcastDims = array<i64: 2>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<32x256x16xi32>
// CHECK:           [[VAR_9_:%.+]] = tensor.empty() : tensor<256xi32>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_9_]] : tensor<256xi32>) {
// CHECK:           ^bb0([[out_]]: i32):
// CHECK:             [[VAR_29_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_30_1_:%.+]] = arith.index_cast [[VAR_29_2_]] : index to i32
// CHECK:             linalg.yield [[VAR_30_1_]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK-DAG:       [[VAR_expanded_1_:%.+]] = tensor.expand_shape [[VAR_10_]] {{.}}[0, 1]{{.}} output_shape [1, 256] : tensor<256xi32> into tensor<1x256xi32>
// CHECK-DAG:       [[VAR_11_:%.+]] = tensor.empty() : tensor<32x256xi32>
// CHECK:           [[VAR_12_:%.+]] = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_1_]] : tensor<1x256xi32>) outs([[VAR_11_]] : tensor<32x256xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<32x256xi32>
// CHECK-DAG:       [[VAR_expanded_2_:%.+]] = tensor.expand_shape [[VAR_12_]] {{.}}[0], [1, 2]{{.}} output_shape [32, 256, 1] : tensor<32x256xi32> into tensor<32x256x1xi32>
// CHECK-DAG:       [[VAR_13_:%.+]] = tensor.empty() : tensor<32x256x16xi32>
// CHECK:           [[VAR_14_:%.+]] = linalg.generic {indexing_maps = [#map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins([[VAR_expanded_2_]] : tensor<32x256x1xi32>) outs([[VAR_13_]] : tensor<32x256x16xi32>) attrs =  {broadcastDims = array<i64: 2>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<32x256x16xi32>
// CHECK:           [[VAR_15_:%.+]] = tensor.empty() : tensor<16xi32>
// CHECK:           [[VAR_16_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_15_]] : tensor<16xi32>) {
// CHECK:           ^bb0([[out_]]: i32):
// CHECK:             [[VAR_29_3_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_30_2_:%.+]] = arith.index_cast [[VAR_29_3_]] : index to i32
// CHECK:             linalg.yield [[VAR_30_2_]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK-DAG:       [[VAR_expanded_3_:%.+]] = tensor.expand_shape [[VAR_16_]] {{.}}[0, 1]{{.}} output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
// CHECK-DAG:       [[VAR_17_:%.+]] = tensor.empty() : tensor<256x16xi32>
// CHECK:           [[VAR_18_:%.+]] = linalg.generic {indexing_maps = [#map5, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_3_]] : tensor<1x16xi32>) outs([[VAR_17_]] : tensor<256x16xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<256x16xi32>
// CHECK-DAG:       [[VAR_expanded_4_:%.+]] = tensor.expand_shape [[VAR_18_]] {{.}}[0, 1], [2]{{.}} output_shape [1, 256, 16] : tensor<256x16xi32> into tensor<1x256x16xi32>
// CHECK-DAG:       [[VAR_19_:%.+]] = tensor.empty() : tensor<32x256x16xi32>
// CHECK:           [[VAR_20_:%.+]] = linalg.generic {indexing_maps = [#map6, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins([[VAR_expanded_4_]] : tensor<1x256x16xi32>) outs([[VAR_19_]] : tensor<32x256x16xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<32x256x16xi32>
// CHECK:           [[VAR_21_:%.+]] = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins([[VAR_8_]], [[VAR_14_]] : tensor<32x256x16xi32>, tensor<32x256x16xi32>) outs([[VAR_8_]] : tensor<32x256x16xi32>) {
// CHECK:           ^bb0([[in_]]: i32, [[in_]]_5: i32, [[out_]]: i32):
// CHECK:             [[VAR_29_4_:%.+]] = arith.addi [[in_]], [[in_]]_5 : i32
// CHECK:             linalg.yield [[VAR_29_4_]] : i32
// CHECK:           } -> tensor<32x256x16xi32>
// CHECK:           [[VAR_22_:%.+]] = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins([[VAR_21_]], [[VAR_20_]] : tensor<32x256x16xi32>, tensor<32x256x16xi32>) outs([[VAR_21_]] : tensor<32x256x16xi32>) {
// CHECK:           ^bb0([[in_]]: i32, [[in_]]_5: i32, [[out_]]: i32):
// CHECK:             [[VAR_29_5_:%.+]] = arith.addi [[in_]], [[in_]]_5 : i32
// CHECK:             linalg.yield [[VAR_29_5_]] : i32
// CHECK:           } -> tensor<32x256x16xi32>
// CHECK:           [[VAR_23_:%.+]] = tensor.empty() : tensor<32x256x16x!tt.ptr<bf16>>
// CHECK:           [[VAR_24_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<bf16>) outs([[VAR_23_]] : tensor<32x256x16x!tt.ptr<bf16>>) -> tensor<32x256x16x!tt.ptr<bf16>>
// CHECK:           [[VAR_25_:%.+]] = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins([[VAR_24_]], [[VAR_22_]] : tensor<32x256x16x!tt.ptr<bf16>>, tensor<32x256x16xi32>) outs([[VAR_24_]] : tensor<32x256x16x!tt.ptr<bf16>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<bf16>, [[in_]]_5: i32, [[out_]]: !tt.ptr<bf16>):
// CHECK:             [[VAR_29_6_:%.+]] = tt.addptr [[in_]], [[in_]]_5 : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield [[VAR_29_6_]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<32x256x16x!tt.ptr<bf16>>
// CHECK-DAG:       [[LOAD_VAR_25_MEM_:%.+]] = tt.load [[VAR_25_]] : tensor<32x256x16x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_27_:%.+]] = tensor.empty() : tensor<256x16xbf16>
// CHECK:           [[VAR_28_:%.+]] = linalg.fill ins([[CST_0_]] : bf16) outs([[VAR_27_]] : tensor<256x16xbf16>) -> tensor<256x16xbf16>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[LOAD_VAR_25_MEM_]] : tensor<32x256x16xbf16>) outs([[VAR_28_]] : tensor<256x16xbf16>) dimensions = [0]
// CHECK:             ([[in_]]: bf16, [[in_]]it: bf16) {
// CHECK:               [[VAR_29_7_:%.+]] = arith.maximumf [[in_]], [[in_]]it : bf16
// CHECK:               linalg.yield [[VAR_29_7_]] : bf16
// CHECK:             }
// CHECK:           tt.store [[PARAM_1_]], [[VAR_reduced_]] : tensor<256x16x!tt.ptr<bf16>>
// CHECK:           return
// CHECK:         }
