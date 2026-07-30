// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s
module {
  tt.func @kernel(
    %f32ptr : !tt.ptr<f32>,
    %intptr : !tt.ptr<i32>,
    %f16ptr : !tt.ptr<f16>,
    %save0 : tensor<1024x!tt.ptr<bf16>>,
    %save1 : tensor<1024x!tt.ptr<f32>>,
    %save2 : tensor<1024x!tt.ptr<f32>>,
    %save3 : tensor<1024x!tt.ptr<f32>>,
    %save4 : tensor<1024x!tt.ptr<f32>>
  ) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // f32ptr pointer
    %8 = tt.splat %f32ptr : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // intptr pointer
    %18 = tt.splat %intptr : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
    // f32ptr pointer
    %28 = tt.splat %f16ptr : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %29 = tt.addptr %28, %0 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %afm = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    %aim = tt.load %19 : tensor<1024x!tt.ptr<i32>>
    %bfm = tt.load %29 : tensor<1024x!tt.ptr<f16>>
    %5 = arith.truncf %afm : tensor<1024xf32> to tensor<1024xbf16>
    %6 = math.exp %afm : tensor<1024xf32>
    %7 = arith.sitofp %aim : tensor<1024xi32> to tensor<1024xf32>
    %10 = arith.extf %bfm : tensor<1024xf16> to tensor<1024xf32>
    %11 = math.sqrt %afm : tensor<1024xf32>
    tt.store %save0, %5 : tensor<1024x!tt.ptr<bf16>>
    tt.store %save1, %6 : tensor<1024x!tt.ptr<f32>>
    tt.store %save2, %7 : tensor<1024x!tt.ptr<f32>>
    tt.store %save3, %10 : tensor<1024x!tt.ptr<f32>>
    tt.store %save4, %11 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<i32>, [[PARAM_2_:%.+]]: !tt.ptr<f16>, [[PARAM_3_:%.+]]: tensor<1024x!tt.ptr<bf16>>, [[PARAM_4_:%.+]]: tensor<1024x!tt.ptr<f32>>, [[PARAM_5_:%.+]]: tensor<1024x!tt.ptr<f32>>, [[PARAM_6_:%.+]]: tensor<1024x!tt.ptr<f32>>, [[PARAM_7_:%.+]]: tensor<1024x!tt.ptr<f32>>, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<1024xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<1024xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_22_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_23_:%.+]] = arith.index_cast [[VAR_22_]] : index to i32
// CHECK:             linalg.yield [[VAR_23_]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           [[VAR_2_:%.+]] = tensor.empty() : tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<f32>) outs([[VAR_2_]] : tensor<1024x!tt.ptr<f32>>) -> tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_3_]], [[VAR_1_]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs([[VAR_3_]] : tensor<1024x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_:%.+]]: !tt.ptr<f32>, [[in_]]_0: i32, [[out_:%.+]]: !tt.ptr<f32>):
// CHECK:             [[VAR_22_1_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_22_1_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_5_:%.+]] = tensor.empty() : tensor<1024x!tt.ptr<i32>>
// CHECK:           [[VAR_6_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<i32>) outs([[VAR_5_]] : tensor<1024x!tt.ptr<i32>>) -> tensor<1024x!tt.ptr<i32>>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_6_]], [[VAR_1_]] : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>) outs([[VAR_6_]] : tensor<1024x!tt.ptr<i32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<i32>, [[in_]]_0: i32, [[out_]]: !tt.ptr<i32>):
// CHECK:             [[VAR_22_2_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<i32>, i32
// CHECK:             linalg.yield [[VAR_22_2_]] : !tt.ptr<i32>
// CHECK:           } -> tensor<1024x!tt.ptr<i32>>
// CHECK:           [[VAR_8_:%.+]] = tensor.empty() : tensor<1024x!tt.ptr<f16>>
// CHECK:           [[VAR_9_:%.+]] = linalg.fill ins([[PARAM_2_]] : !tt.ptr<f16>) outs([[VAR_8_]] : tensor<1024x!tt.ptr<f16>>) -> tensor<1024x!tt.ptr<f16>>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_9_]], [[VAR_1_]] : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>) outs([[VAR_9_]] : tensor<1024x!tt.ptr<f16>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f16>, [[in_]]_0: i32, [[out_]]: !tt.ptr<f16>):
// CHECK:             [[VAR_22_3_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<f16>, i32
// CHECK:             linalg.yield [[VAR_22_3_]] : !tt.ptr<f16>
// CHECK:           } -> tensor<1024x!tt.ptr<f16>>
// CHECK-DAG:       [[LOAD_VAR_4_MEM_:%.+]] = tt.load [[VAR_4_]] : tensor<1024x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_7_MEM_:%.+]] = tt.load [[VAR_7_]] : tensor<1024x!tt.ptr<i32>>
// CHECK-DAG:       [[LOAD_VAR_10_MEM_:%.+]] = tt.load [[VAR_10_]] : tensor<1024x!tt.ptr<f16>>
// CHECK-DAG:       [[VAR_14_:%.+]] = tensor.empty() : tensor<1024xbf16>
// CHECK:           [[VAR_15_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[LOAD_VAR_4_MEM_]] : tensor<1024xf32>) outs([[VAR_14_]] : tensor<1024xbf16>) {
// CHECK:           ^bb0([[in_]]: f32, [[out_]]: bf16):
// CHECK:             [[VAR_22_4_:%.+]] = arith.truncf [[in_]] : f32 to bf16
// CHECK:             linalg.yield [[VAR_22_4_]] : bf16
// CHECK:           } -> tensor<1024xbf16>
// CHECK:           [[VAR_16_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[LOAD_VAR_4_MEM_]] : tensor<1024xf32>) outs([[LOAD_VAR_4_MEM_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_]]: f32, [[out_]]: f32):
// CHECK:             [[VAR_22_5_:%.+]] = math.exp [[in_]] : f32
// CHECK:             linalg.yield [[VAR_22_5_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_17_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:           [[VAR_18_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[LOAD_VAR_7_MEM_]] : tensor<1024xi32>) outs([[VAR_17_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: f32):
// CHECK:             [[VAR_22_6_:%.+]] = arith.sitofp [[in_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_22_6_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_19_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:           [[VAR_20_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[LOAD_VAR_10_MEM_]] : tensor<1024xf16>) outs([[VAR_19_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_]]: f16, [[out_]]: f32):
// CHECK:             [[VAR_22_7_:%.+]] = arith.extf [[in_]] : f16 to f32
// CHECK:             linalg.yield [[VAR_22_7_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_21_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[LOAD_VAR_4_MEM_]] : tensor<1024xf32>) outs([[LOAD_VAR_4_MEM_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_]]: f32, [[out_]]: f32):
// CHECK:             [[VAR_22_8_:%.+]] = math.sqrt [[in_]] : f32
// CHECK:             linalg.yield [[VAR_22_8_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           tt.store [[PARAM_3_]], [[VAR_15_]] : tensor<1024x!tt.ptr<bf16>>
// CHECK:           tt.store [[PARAM_4_]], [[VAR_16_]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.store [[PARAM_5_]], [[VAR_18_]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.store [[PARAM_6_]], [[VAR_20_]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           tt.store [[PARAM_7_]], [[VAR_21_]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
