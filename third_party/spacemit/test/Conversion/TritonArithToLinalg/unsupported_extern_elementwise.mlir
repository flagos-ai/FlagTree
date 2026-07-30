// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @rand(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    %3 = tt.load %2 : tensor<8x!tt.ptr<i32>>
    %4 = tt.extern_elementwise %3, %0 {libname = "", libpath = "", pure = true, symbol = "some_symbol"} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
    %5 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %6 = tt.addptr %5, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
    tt.store %6, %4 : tensor<8x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @rand
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<i32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<8xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<8xi32>) {
// CHECK:           ^bb0([[out_:%.*]]: i32):
// CHECK:             [[VAR_10_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i32
// CHECK:             linalg.yield [[VAR_11_]] : i32
// CHECK:           } -> tensor<8xi32>
// CHECK:           [[VAR_2_:%.+]] = tensor.empty() : tensor<8x!tt.ptr<i32>>
// CHECK:           [[VAR_3_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<i32>) outs([[VAR_2_]] : tensor<8x!tt.ptr<i32>>) -> tensor<8x!tt.ptr<i32>>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_3_]], [[VAR_1_]] : tensor<8x!tt.ptr<i32>>, tensor<8xi32>) outs([[VAR_3_]] : tensor<8x!tt.ptr<i32>>) {
// CHECK:           ^bb0([[in_:%.+]]: !tt.ptr<i32>, [[in_]]_0: i32, [[out_:%.+]]: !tt.ptr<i32>):
// CHECK:             [[VAR_10_1_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<i32>, i32
// CHECK:             linalg.yield [[VAR_10_1_]] : !tt.ptr<i32>
// CHECK:           } -> tensor<8x!tt.ptr<i32>>
// CHECK:           [[LOAD_VAR_4_MEM_:%.+]] = tt.load [[VAR_4_]] : tensor<8x!tt.ptr<i32>>
// CHECK-DAG:       [[VAR_6_:%.+]] = tt.extern_elementwise [[LOAD_VAR_4_MEM_]], [[VAR_1_]] {libname = "", libpath = "", pure = true, symbol = "some_symbol"} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tensor.empty() : tensor<8x!tt.ptr<i32>>
// CHECK:           [[VAR_8_:%.+]] = linalg.fill ins([[PARAM_1_]] : !tt.ptr<i32>) outs([[VAR_7_]] : tensor<8x!tt.ptr<i32>>) -> tensor<8x!tt.ptr<i32>>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_8_]], [[VAR_1_]] : tensor<8x!tt.ptr<i32>>, tensor<8xi32>) outs([[VAR_8_]] : tensor<8x!tt.ptr<i32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<i32>, [[in_]]_0: i32, [[out_]]: !tt.ptr<i32>):
// CHECK:             [[VAR_10_2_:%.+]] = tt.addptr [[in_]], [[in_]]_0 : !tt.ptr<i32>, i32
// CHECK:             linalg.yield [[VAR_10_2_]] : !tt.ptr<i32>
// CHECK:           } -> tensor<8x!tt.ptr<i32>>
// CHECK:           tt.store [[VAR_9_]], [[VAR_6_]] : tensor<8x!tt.ptr<i32>>
// CHECK:           return
// CHECK:         }
