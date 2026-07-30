// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>
  )
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c64 = arith.constant 128 : i32
    %1 = tt.splat %c64 : i32 -> tensor<128xi32>
    %2 = arith.muli %0, %1 : tensor<128xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %4 = tt.broadcast %3 : tensor<128x1xi32> -> tensor<128x64xi32>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %7 = tt.broadcast %6 : tensor<1x64xi32> -> tensor<128x64xi32>
    %8 = arith.addi %4, %7 : tensor<128x64xi32>
    %10 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %11 = tt.expand_dims %10 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
    %12 = tt.broadcast %11 : tensor<256x1xi32> -> tensor<256x64xi32>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %c256 = arith.constant 256 : i32
    %14 = tt.splat %c256 : i32 -> tensor<64xi32>
    %15 = arith.muli %13, %14 : tensor<64xi32>
    %16 = tt.expand_dims %15 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %17 = tt.broadcast %16 : tensor<1x64xi32> -> tensor<256x64xi32>
    %18 = arith.addi %12, %17 : tensor<256x64xi32>
    %20 = tt.splat %c256 : i32 -> tensor<128xi32>
    %21 = arith.muli %0, %20 : tensor<128xi32>
    %22 = tt.expand_dims %21 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %23 = tt.broadcast %22 : tensor<128x1xi32> -> tensor<128x256xi32>
    %24 = tt.expand_dims %10 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %25 = tt.broadcast %24 {axis = 0 : i32} : tensor<1x256xi32> -> tensor<128x256xi32>
    %26 = arith.addi %23, %25 : tensor<128x256xi32>
    %30 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    %31 = tt.addptr %30, %8 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    %32 = tt.load %31 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<128x64x!tt.ptr<bf16>>
    %40 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<256x64x!tt.ptr<bf16>>
    %41 = tt.addptr %40, %18 : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>
    %42 = tt.load %41 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<256x64x!tt.ptr<bf16>>
    %43 = tt.trans %42 {order = array<i32: 1, 0>} : tensor<256x64xbf16> -> tensor<64x256xbf16>
    %50 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
    %51 = tt.addptr %50, %26 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %52 = tt.load %51 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<128x256x!tt.ptr<bf16>>
    %60 = tt.dot %32, %43, %52 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
    tt.store %51, %60 : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d0, 0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: !tt.ptr<bf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0.000000e+00 : bf16
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<128xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_256_]] : i32) outs([[VAR_0_]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tensor.empty() : tensor<64xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = linalg.fill ins([[CST_256_]] : i32) outs([[VAR_2_]] : tensor<64xi32>) -> tensor<64xi32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tensor.empty() : tensor<128xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = linalg.fill ins([[CST_128_]] : i32) outs([[VAR_4_]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK-DAG:       [[VAR_6_:%.+]] = tensor.empty() : tensor<128xi32>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_6_]] : tensor<128xi32>) {
// CHECK:           ^bb0([[out_:.+]]: i32):
// CHECK:             [[VAR_49_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_50_:%.+]] = arith.index_cast [[VAR_49_]] : index to i32
// CHECK:             linalg.yield [[VAR_50_]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]], [[VAR_5_]] : tensor<128xi32>, tensor<128xi32>) outs([[VAR_7_]] : tensor<128xi32>) {
// CHECK:           ^bb0([[in_:.+]]: i32, [[in_1:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             [[VAR_49_1_:%.+]] = arith.muli [[in_]], [[in_1:.+]] : i32
// CHECK:             linalg.yield [[VAR_49_1_]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK-DAG:       [[VAR_expanded_:%.+]] = tensor.expand_shape [[VAR_8_]] {{.}}[0, 1]{{.}} output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tensor.empty() : tensor<128x64xi32>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_]] : tensor<128x1xi32>) outs([[VAR_9_]] : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0([[in_:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<128x64xi32>
// CHECK:           [[VAR_11_:%.+]] = tensor.empty() : tensor<64xi32>
// CHECK:           [[VAR_12_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_11_]] : tensor<64xi32>) {
// CHECK:           ^bb0([[out_:.+]]: i32):
// CHECK:             [[VAR_49_2_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_50_1_:%.+]] = arith.index_cast [[VAR_49_2_]] : index to i32
// CHECK:             linalg.yield [[VAR_50_1_]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK-DAG:       [[VAR_expanded_1_:%.+]] = tensor.expand_shape [[VAR_12_]] {{.}}[0, 1]{{.}} output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
// CHECK-DAG:       [[VAR_13_:%.+]] = tensor.empty() : tensor<128x64xi32>
// CHECK:           [[VAR_14_:%.+]] = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_1_]] : tensor<1x64xi32>) outs([[VAR_13_]] : tensor<128x64xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<128x64xi32>
// CHECK:           [[VAR_15_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_10_]], [[VAR_14_]] : tensor<128x64xi32>, tensor<128x64xi32>) outs([[VAR_10_]] : tensor<128x64xi32>) {
// CHECK:           ^bb0([[in_:.+]]: i32, [[in_1:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             [[VAR_49_3_:%.+]] = arith.addi [[in_]], [[in_1:.+]] : i32
// CHECK:             linalg.yield [[VAR_49_3_]] : i32
// CHECK:           } -> tensor<128x64xi32>
// CHECK:           [[VAR_16_:%.+]] = tensor.empty() : tensor<256xi32>
// CHECK:           [[VAR_17_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_16_]] : tensor<256xi32>) {
// CHECK:           ^bb0([[out_:.+]]: i32):
// CHECK:             [[VAR_49_4_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_50_2_:%.+]] = arith.index_cast [[VAR_49_4_]] : index to i32
// CHECK:             linalg.yield [[VAR_50_2_]] : i32
// CHECK:           } -> tensor<256xi32>
// CHECK-DAG:       [[VAR_expanded_2_:%.+]] = tensor.expand_shape [[VAR_17_]] {{.}}[0, 1]{{.}} output_shape [256, 1] : tensor<256xi32> into tensor<256x1xi32>
// CHECK-DAG:       [[VAR_18_:%.+]] = tensor.empty() : tensor<256x64xi32>
// CHECK:           [[VAR_19_:%.+]] = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_2_]] : tensor<256x1xi32>) outs([[VAR_18_]] : tensor<256x64xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0([[in_:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<256x64xi32>
// CHECK:           [[VAR_20_:%.+]] = tensor.empty() : tensor<64xi32>
// CHECK:           [[VAR_21_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_20_]] : tensor<64xi32>) {
// CHECK:           ^bb0([[out_:.+]]: i32):
// CHECK:             [[VAR_49_5_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_50_3_:%.+]] = arith.index_cast [[VAR_49_5_]] : index to i32
// CHECK:             linalg.yield [[VAR_50_3_]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK:           [[VAR_22_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_21_]], [[VAR_3_]] : tensor<64xi32>, tensor<64xi32>) outs([[VAR_21_]] : tensor<64xi32>) {
// CHECK:           ^bb0([[in_:.+]]: i32, [[in_1:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             [[VAR_49_6_:%.+]] = arith.muli [[in_]], [[in_1:.+]] : i32
// CHECK:             linalg.yield [[VAR_49_6_]] : i32
// CHECK:           } -> tensor<64xi32>
// CHECK-DAG:       [[VAR_expanded_3_:%.+]] = tensor.expand_shape [[VAR_22_]] {{.}}[0, 1]{{.}} output_shape [1, 64] : tensor<64xi32> into tensor<1x64xi32>
// CHECK-DAG:       [[VAR_23_:%.+]] = tensor.empty() : tensor<256x64xi32>
// CHECK:           [[VAR_24_:%.+]] = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_3_]] : tensor<1x64xi32>) outs([[VAR_23_]] : tensor<256x64xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<256x64xi32>
// CHECK:           [[VAR_25_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_19_]], [[VAR_24_]] : tensor<256x64xi32>, tensor<256x64xi32>) outs([[VAR_19_]] : tensor<256x64xi32>) {
// CHECK:           ^bb0([[in_:.+]]: i32, [[in_1:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             [[VAR_49_7_:%.+]] = arith.addi [[in_]], [[in_1:.+]] : i32
// CHECK:             linalg.yield [[VAR_49_7_]] : i32
// CHECK:           } -> tensor<256x64xi32>
// CHECK:           [[VAR_26_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]], [[VAR_1_]] : tensor<128xi32>, tensor<128xi32>) outs([[VAR_7_]] : tensor<128xi32>) {
// CHECK:           ^bb0([[in_:.+]]: i32, [[in_1:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             [[VAR_49_8_:%.+]] = arith.muli [[in_]], [[in_1:.+]] : i32
// CHECK:             linalg.yield [[VAR_49_8_]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK-DAG:       [[VAR_expanded_4_:%.+]] = tensor.expand_shape [[VAR_26_]] {{.}}[0, 1]{{.}} output_shape [128, 1] : tensor<128xi32> into tensor<128x1xi32>
// CHECK-DAG:       [[VAR_27_:%.+]] = tensor.empty() : tensor<128x256xi32>
// CHECK:           [[VAR_28_:%.+]] = linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_4_]] : tensor<128x1xi32>) outs([[VAR_27_]] : tensor<128x256xi32>) attrs =  {broadcastDims = array<i64: 1>} {
// CHECK:           ^bb0([[in_:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<128x256xi32>
// CHECK-DAG:       [[VAR_expanded_5_:%.+]] = tensor.expand_shape [[VAR_17_]] {{.}}[0, 1]{{.}} output_shape [1, 256] : tensor<256xi32> into tensor<1x256xi32>
// CHECK-DAG:       [[VAR_29_:%.+]] = tensor.empty() : tensor<128x256xi32>
// CHECK:           [[VAR_30_:%.+]] = linalg.generic {indexing_maps = [#map3, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_expanded_5_]] : tensor<1x256xi32>) outs([[VAR_29_]] : tensor<128x256xi32>) attrs =  {broadcastDims = array<i64: 0>} {
// CHECK:           ^bb0([[in_:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             linalg.yield [[in_]] : i32
// CHECK:           } -> tensor<128x256xi32>
// CHECK:           [[VAR_31_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_28_]], [[VAR_30_]] : tensor<128x256xi32>, tensor<128x256xi32>) outs([[VAR_28_]] : tensor<128x256xi32>) {
// CHECK:           ^bb0([[in_:.+]]: i32, [[in_1:.+]]: i32, [[out_:.+]]: i32):
// CHECK:             [[VAR_49_9_:%.+]] = arith.addi [[in_]], [[in_1:.+]] : i32
// CHECK:             linalg.yield [[VAR_49_9_]] : i32
// CHECK:           } -> tensor<128x256xi32>
// CHECK:           [[VAR_32_:%.+]] = tensor.empty() : tensor<128x64x!tt.ptr<bf16>>
// CHECK:           [[VAR_33_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<bf16>) outs([[VAR_32_]] : tensor<128x64x!tt.ptr<bf16>>) -> tensor<128x64x!tt.ptr<bf16>>
// CHECK:           [[VAR_34_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_33_]], [[VAR_15_]] : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>) outs([[VAR_33_]] : tensor<128x64x!tt.ptr<bf16>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<bf16>, [[in_1:.+]]: i32, [[out_]]: !tt.ptr<bf16>):
// CHECK:             [[VAR_49_10_:%.+]] = tt.addptr [[in_]], [[in_1]] : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield [[VAR_49_10_]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<128x64x!tt.ptr<bf16>>
// CHECK-DAG:       [[LOAD_VAR_34_MEM_:%.+]] = tt.load [[VAR_34_]] : tensor<128x64x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_36_:%.+]] = tensor.empty() : tensor<256x64x!tt.ptr<bf16>>
// CHECK:           [[VAR_37_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<bf16>) outs([[VAR_36_]] : tensor<256x64x!tt.ptr<bf16>>) -> tensor<256x64x!tt.ptr<bf16>>
// CHECK:           [[VAR_38_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_37_]], [[VAR_25_]] : tensor<256x64x!tt.ptr<bf16>>, tensor<256x64xi32>) outs([[VAR_37_]] : tensor<256x64x!tt.ptr<bf16>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<bf16>, [[in_1:.+]]: i32, [[out_]]: !tt.ptr<bf16>):
// CHECK:             [[VAR_49_11_:%.+]] = tt.addptr [[in_]], [[in_1]] : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield [[VAR_49_11_]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<256x64x!tt.ptr<bf16>>
// CHECK-DAG:       [[LOAD_VAR_38_MEM_:%.+]] = tt.load [[VAR_38_]] : tensor<256x64x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_40_:%.+]] = tensor.empty() : tensor<64x256xbf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_transposed_:%.+]] = linalg.transpose ins([[LOAD_VAR_38_MEM_]] : tensor<256x64xbf16>) outs([[VAR_40_]] : tensor<64x256xbf16>) permutation = [1, 0]
// CHECK-DAG:       [[VAR_41_:%.+]] = tensor.empty() : tensor<128x256x!tt.ptr<bf16>>
// CHECK:           [[VAR_42_:%.+]] = linalg.fill ins([[PARAM_2_]] : !tt.ptr<bf16>) outs([[VAR_41_]] : tensor<128x256x!tt.ptr<bf16>>) -> tensor<128x256x!tt.ptr<bf16>>
// CHECK:           [[VAR_43_:%.+]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins([[VAR_42_]], [[VAR_31_]] : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>) outs([[VAR_42_]] : tensor<128x256x!tt.ptr<bf16>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<bf16>, [[in_1:.+]]: i32, [[out_]]: !tt.ptr<bf16>):
// CHECK:             [[VAR_49_12_:%.+]] = tt.addptr [[in_]], [[in_1]] : !tt.ptr<bf16>, i32
// CHECK:             linalg.yield [[VAR_49_12_]] : !tt.ptr<bf16>
// CHECK:           } -> tensor<128x256x!tt.ptr<bf16>>
// CHECK-DAG:       [[LOAD_VAR_43_MEM_:%.+]] = tt.load [[VAR_43_]] : tensor<128x256x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_45_:%.+]] = tensor.empty() : tensor<128x256xbf16>
// CHECK:           [[VAR_46_:%.+]] = linalg.fill ins([[CST_0_]] : bf16) outs([[VAR_45_]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
// CHECK:           [[VAR_47_:%.+]] = linalg.matmul ins([[LOAD_VAR_34_MEM_]], [[VAR_transposed_]] : tensor<128x64xbf16>, tensor<64x256xbf16>) outs([[VAR_46_]] : tensor<128x256xbf16>) -> tensor<128x256xbf16>
// CHECK:           [[VAR_48_:%.+]] = linalg.generic {indexing_maps = [[[MAP_2_]], [[MAP_2_]], [[MAP_2_]]], iterator_types = ["parallel", "parallel"]} ins([[LOAD_VAR_43_MEM_]], [[VAR_47_]] : tensor<128x256xbf16>, tensor<128x256xbf16>) outs([[LOAD_VAR_43_MEM_]] : tensor<128x256xbf16>) {
// CHECK:           ^bb0([[VAR_in_1:%.+]]: bf16, [[VAR_in_2:%.+]]: bf16, {{%.+}}: bf16):
// CHECK:             [[VAR_49_:%.+]] = arith.addf [[VAR_in_1]], [[VAR_in_2]] : bf16
// CHECK:             linalg.yield [[VAR_49_]] : bf16
// CHECK:           } -> tensor<128x256xbf16>
// CHECK:           tt.store [[VAR_43_]], [[VAR_48_]] : tensor<128x256x!tt.ptr<bf16>>
// CHECK:           return
// CHECK:         }
