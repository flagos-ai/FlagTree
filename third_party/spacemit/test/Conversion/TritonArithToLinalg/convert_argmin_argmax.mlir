// RUN: triton-shared-opt --triton-arith-to-linalg --split-input-file %s | FileCheck %s

module {
  tt.func public @argmax_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %3 = tt.splat %1 : i32 -> tensor<4096xi32>
    %4 = arith.addi %3, %2 : tensor<4096xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    %7 = tt.load %6 : tensor<4096x!tt.ptr<f32>>
    %8:2 = "tt.reduce"(%7, %2) <{axis = 0 : i32}> ({
    ^bb0(%arg9: f32, %arg10: i32, %arg11: f32, %arg12: i32):
      %11 = arith.cmpf oeq, %arg9, %arg11 : f32
      %12 = arith.cmpi slt, %arg10, %arg12 : i32
      %13 = arith.andi %11, %12 : i1
      %14 = arith.cmpf ogt, %arg9, %arg11 : f32
      %15 = arith.ori %14, %13 : i1
      %16 = arith.select %15, %arg9, %arg11 : f32
      %17 = arith.select %15, %arg10, %arg12 : i32
      tt.reduce.return %16, %17 : f32, i32
  }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
    %9 = tt.addptr %arg1, %0 : !tt.ptr<i32>, i32
    tt.store %9, %8#1 : !tt.ptr<i32>
    tt.return
  }
}


// -----

module {
  tt.func public @argmin_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %3 = tt.splat %1 : i32 -> tensor<4096xi32>
    %4 = arith.addi %3, %2 : tensor<4096xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    %7 = tt.load %6 : tensor<4096x!tt.ptr<f32>>
    %8:2 = "tt.reduce"(%7, %2) <{axis = 0 : i32}> ({
    ^bb0(%arg9: f32, %arg10: i32, %arg11: f32, %arg12: i32):
      %11 = arith.cmpf oeq, %arg9, %arg11 : f32
      %12 = arith.cmpi slt, %arg10, %arg12 : i32
      %13 = arith.andi %11, %12 : i1
      %14 = arith.cmpf olt, %arg9, %arg11 : f32
      %15 = arith.ori %14, %13 : i1
      %16 = arith.select %15, %arg9, %arg11 : f32
      %17 = arith.select %15, %arg10, %arg12 : i32
      tt.reduce.return %16, %17 : f32, i32
  }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
    %9 = tt.addptr %arg1, %0 : !tt.ptr<i32>, i32
    tt.store %9, %8#1 : !tt.ptr<i32>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @argmax_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<i32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[PARAM_2_]] : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_1_]] : tensor<4096xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_15_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_16_:%.+]] = arith.index_cast [[VAR_15_]] : index to i32
// CHECK:             linalg.yield [[VAR_16_]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           [[VAR_3_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK:           [[VAR_4_:%.+]] = linalg.fill ins([[VAR_0_]] : i32) outs([[VAR_3_]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_4_]], [[VAR_2_]] : tensor<4096xi32>, tensor<4096xi32>) outs([[VAR_4_]] : tensor<4096xi32>) {
// CHECK:           ^bb0([[in_:%.+]]: i32, [[in_1:%.+]]: i32, [[out_]]: i32):
// CHECK:             [[VAR_15_1_:%.+]] = arith.addi [[in_]], [[in_1]] : i32
// CHECK:             linalg.yield [[VAR_15_1_]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           [[VAR_6_:%.+]] = tensor.empty() : tensor<4096x!tt.ptr<f32>>
// CHECK:           [[VAR_7_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<f32>) outs([[VAR_6_]] : tensor<4096x!tt.ptr<f32>>) -> tensor<4096x!tt.ptr<f32>>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]], [[VAR_5_]] : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>) outs([[VAR_7_]] : tensor<4096x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_1:%.+]]: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_15_2_:%.+]] = tt.addptr [[in_]], [[in_1]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_15_2_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4096x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_8_MEM_:%.+]] = tt.load [[VAR_8_]] : tensor<4096x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_10_:%.+]] = tensor.empty() : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = linalg.fill ins([[CST_0_]] : f32) outs([[VAR_10_]] : tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_12_:%.+]] = tensor.empty() : tensor<i32>
// CHECK:           [[VAR_13_:%.+]] = linalg.fill ins([[CST_minus_1_]] : i32) outs([[VAR_12_]] : tensor<i32>) -> tensor<i32>
// CHECK:           [[VAR_reduced_:%.+]]:2 = linalg.reduce ins([[LOAD_VAR_8_MEM_]], [[VAR_2_]] : tensor<4096xf32>, tensor<4096xi32>) outs([[VAR_11_]], [[VAR_13_]] : tensor<f32>, tensor<i32>) dimensions = [0]
// CHECK:             ([[in_:%.*]]: f32, [[in_1_:%.*]]: i32, [[init_:%.*]]: f32, [[init_2_:%.*]]: i32) {
// CHECK-DAG:           [[VAR_15_3_:%.+]] = arith.cmpf oeq, [[in_]], [[init_]] : f32
// CHECK-DAG:           [[VAR_16_1_:%.+]] = arith.cmpi slt, [[in_1_]], [[init_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_17_:%.+]] = arith.andi [[VAR_15_3_]], [[VAR_16_1_]] : i1
// CHECK-DAG:           [[VAR_18_:%.+]] = arith.cmpf ogt, [[in_]], [[init_]] : f32
// CHECK:               [[VAR_19_:%.+]] = arith.ori [[VAR_18_]], [[VAR_17_]] : i1
// CHECK-DAG:           [[VAR_20_:%.+]] = arith.select [[VAR_19_]], [[in_]], [[init_]] : f32
// CHECK-DAG:           [[VAR_21_:%.+]] = arith.select [[VAR_19_]], [[in_1_]], [[init_2_]] : i32
// CHECK:               linalg.yield [[VAR_20_]], [[VAR_21_]] : f32, i32
// CHECK:             }
// CHECK-DAG:       [[VAR_extracted_0_:%.+]] = tensor.extract [[VAR_reduced_]]#1[] : tensor<i32>
// CHECK-DAG:       [[VAR_14_:%.+]] = tt.addptr [[PARAM_1_]], [[PARAM_6_]] : !tt.ptr<i32>, i32
// CHECK:           tt.store [[VAR_14_]], [[VAR_extracted_0_]] : !tt.ptr<i32>
// CHECK:           return
// CHECK:         }

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @argmin_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<i32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[PARAM_2_]] : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_1_]] : tensor<4096xi32>) {
// CHECK:           ^bb0([[out_:%.+]]: i32):
// CHECK:             [[VAR_15_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_16_:%.+]] = arith.index_cast [[VAR_15_]] : index to i32
// CHECK:             linalg.yield [[VAR_16_]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           [[VAR_3_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK:           [[VAR_4_:%.+]] = linalg.fill ins([[VAR_0_]] : i32) outs([[VAR_3_]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_4_]], [[VAR_2_]] : tensor<4096xi32>, tensor<4096xi32>) outs([[VAR_4_]] : tensor<4096xi32>) {
// CHECK:           ^bb0([[in_:%.+]]: i32, [[in_1:%.+]]: i32, [[out_]]: i32):
// CHECK:             [[VAR_15_1_:%.+]] = arith.addi [[in_]], [[in_1]] : i32
// CHECK:             linalg.yield [[VAR_15_1_]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           [[VAR_6_:%.+]] = tensor.empty() : tensor<4096x!tt.ptr<f32>>
// CHECK:           [[VAR_7_:%.+]] = linalg.fill ins([[PARAM_0_]] : !tt.ptr<f32>) outs([[VAR_6_]] : tensor<4096x!tt.ptr<f32>>) -> tensor<4096x!tt.ptr<f32>>
// CHECK:           [[VAR_8_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_7_]], [[VAR_5_]] : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>) outs([[VAR_7_]] : tensor<4096x!tt.ptr<f32>>) {
// CHECK:           ^bb0([[in_]]: !tt.ptr<f32>, [[in_1:%.+]]: i32, [[out_]]: !tt.ptr<f32>):
// CHECK:             [[VAR_15_2_:%.+]] = tt.addptr [[in_]], [[in_1]] : !tt.ptr<f32>, i32
// CHECK:             linalg.yield [[VAR_15_2_]] : !tt.ptr<f32>
// CHECK:           } -> tensor<4096x!tt.ptr<f32>>
// CHECK-DAG:       [[LOAD_VAR_8_MEM_:%.+]] = tt.load [[VAR_8_]] : tensor<4096x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_10_:%.+]] = tensor.empty() : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = linalg.fill ins([[CST_0_]] : f32) outs([[VAR_10_]] : tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_12_:%.+]] = tensor.empty() : tensor<i32>
// CHECK:           [[VAR_13_:%.+]] = linalg.fill ins([[CST_minus_1_]] : i32) outs([[VAR_12_]] : tensor<i32>) -> tensor<i32>
// CHECK:           [[VAR_reduced_:%.+]]:2 = linalg.reduce ins([[LOAD_VAR_8_MEM_]], [[VAR_2_]] : tensor<4096xf32>, tensor<4096xi32>) outs([[VAR_11_]], [[VAR_13_]] : tensor<f32>, tensor<i32>) dimensions = [0]
// CHECK:             ([[in_]]: f32, [[in_1_]]: i32, [[init_]]: f32, [[init_2_]]: i32) {
// CHECK-DAG:           [[VAR_15_3_:%.+]] = arith.cmpf oeq, [[in_]], [[in_]]it : f32
// CHECK-DAG:           [[VAR_16_1_:%.+]] = arith.cmpi slt, [[in_1_]], [[init_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_17_:%.+]] = arith.andi [[VAR_15_3_]], [[VAR_16_1_]] : i1
// CHECK-DAG:           [[VAR_18_:%.+]] = arith.cmpf olt, [[in_]], [[in_]]it : f32
// CHECK:               [[VAR_19_:%.+]] = arith.ori [[VAR_18_]], [[VAR_17_]] : i1
// CHECK-DAG:           [[VAR_20_:%.+]] = arith.select [[VAR_19_]], [[in_]], [[in_]]it : f32
// CHECK-DAG:           [[VAR_21_:%.+]] = arith.select [[VAR_19_]], [[in_1_]], [[init_2_]] : i32
// CHECK:               linalg.yield [[VAR_20_]], [[VAR_21_]] : f32, i32
// CHECK:             }
// CHECK-DAG:       [[VAR_extracted_0_:%.+]] = tensor.extract [[VAR_reduced_]]#1[] : tensor<i32>
// CHECK-DAG:       [[VAR_14_:%.+]] = tt.addptr [[PARAM_1_]], [[PARAM_6_]] : !tt.ptr<i32>, i32
// CHECK:           tt.store [[VAR_14_]], [[VAR_extracted_0_]] : !tt.ptr<i32>
// CHECK:           return
// CHECK:         }
