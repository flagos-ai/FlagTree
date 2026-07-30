// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func @kernel(%a : !tt.ptr<i32>, %b : !tt.ptr<f32>) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>

    // a pointer
    %8 = tt.splat %a : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>

    // b pointer
    %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>

    %am = tt.load %9 : tensor<1024x!tt.ptr<i32>>

    // cast result before doing float add
    %am_bitcast = tt.bitcast %am : tensor<1024xi32> -> tensor<1024xf32>


    tt.store %19, %am_bitcast : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}


// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<1024xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<1024xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_11_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_12_:%.+]] = arith.index_cast [[VAR_11_]] : index to i32
// CHECK:             linalg.yield [[VAR_12_]] : i32
// CHECK:           } -> tensor<1024xi32>
// CHECK:           [[VAR_2_:%.+]] = tensor.empty() : tensor<1024x!tt.ptr<i32>>
// CHECK:           [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<i32>) outs([[VAR_2_]] : tensor<1024x!tt.ptr<i32>>) -> tensor<1024x!tt.ptr<i32>>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_3_]], [[VAR_1_]] : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>) outs([[VAR_3_]] : tensor<1024x!tt.ptr<i32>>) {
// CHECK:           ^bb0([[in_:%.+]]: !tt.ptr<i32>, [[in_]]_0: i32, [[out_]]: !tt.ptr<i32>):
// CHECK:             [[VAR_11_1_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<i32>, i32
// CHECK:             linalg.yield [[VAR_11_1_]] : !tt.ptr<i32>
// CHECK:           } -> tensor<1024x!tt.ptr<i32>>
// CHECK:           [[VAR_5_:%.+]] = tensor.empty() : tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_6_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<f32>) outs([[VAR_5_]] : tensor<1024x!tt.ptr<f32>>) -> tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_7_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_6_]], [[VAR_1_]] : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>) outs([[VAR_6_]] : tensor<1024x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_]]_0: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_11_2_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_11_2_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<1024x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_4_MEM_:%.+]] = tt.load [[VAR_4_]] : tensor<1024x!tt.ptr<i32>>
// CHECK-DAG:       [[VAR_9_:%.+]] = tensor.empty() : tensor<1024xf32>
// CHECK:           [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[LOAD_VAR_4_MEM_]] : tensor<1024xi32>) outs([[VAR_9_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[in_]]: i32, [[out_]]: f32):
// CHECK:             [[VAR_11_3_:%.+]] = arith.bitcast [[in_]] : i32 to f32
// CHECK:             linalg.yield [[VAR_11_3_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           tt.store [[VAR_7_]], [[VAR_10_]] : tensor<1024x!tt.ptr<f32>>
// CHECK:           return
// CHECK:         }
