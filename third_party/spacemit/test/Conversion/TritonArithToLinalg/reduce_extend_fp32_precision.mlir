// RUN: triton-shared-opt --triton-to-structured --canonicalize --cse --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @fn1(%arg0: !tt.ptr<f16> , %arg1: !tt.ptr<f16> ) attributes {noinline = false} {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    %3 = tt.load %2 : tensor<32x!tt.ptr<f16>>
    %4 = "tt.reduce"(%3) <{axis = 0 : i32}> ({
    ^bb0(%arg2: f16 , %arg3: f16 ):
      %12 = arith.addf %arg2, %arg3 : f16
      tt.reduce.return %12 : f16
    }) : (tensor<32xf16>) -> f16
    %5 = arith.extf %3 : tensor<32xf16> to tensor<32xf32>
    %6 = arith.extf %4 : f16 to f32
    %7 = tt.splat %6 : f32 -> tensor<32xf32>
    %8 = arith.divf %5, %7 : tensor<32xf32>
    %9 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %10 = tt.addptr %9, %0 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    %11 = arith.truncf %8 : tensor<32xf32> to tensor<32xf16>
    tt.store %10, %11 : tensor<32x!tt.ptr<f16>>
    tt.return
  }

  tt.func public @fn2(%arg0: !tt.ptr<f16> , %arg1: !tt.ptr<f16> ) attributes {noinline = false} {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    %3 = tt.load %2 : tensor<32x!tt.ptr<f16>>
    %4 = "tt.reduce"(%3) <{axis = 0 : i32}> ({
    ^bb0(%arg2: f16 , %arg3: f16 ):
      %9 = arith.addf %arg2, %arg3 : f16
      tt.reduce.return %9 : f16
    }) : (tensor<32xf16>) -> f16
    %5 = tt.splat %4 : f16 -> tensor<32xf16>
    %6 = arith.subf %3, %5 : tensor<32xf16>
    %7 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %8 = tt.addptr %7, %0 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    tt.store %8, %6 : tensor<32x!tt.ptr<f16>>
    tt.return
  }

  tt.func public @fn3(%arg0: !tt.ptr<bf16> , %arg1: !tt.ptr<bf16> ) attributes {noinline = false} {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
    %3 = tt.load %2 : tensor<32x!tt.ptr<bf16>>
    %4 = "tt.reduce"(%3) <{axis = 0 : i32}> ({
    ^bb0(%arg2: bf16 , %arg3: bf16 ):
      %9 = arith.addf %arg2, %arg3 : bf16
      tt.reduce.return %9 : bf16
    }) : (tensor<32xbf16>) -> bf16
    %5 = tt.splat %4 : bf16 -> tensor<32xbf16>
    %6 = arith.subf %3, %5 : tensor<32xbf16>
    %7 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<32x!tt.ptr<bf16>>
    %8 = tt.addptr %7, %0 : tensor<32x!tt.ptr<bf16>>, tensor<32xi32>
    tt.store %8, %6 : tensor<32x!tt.ptr<bf16>>
    tt.return
  }

  tt.func public @fn4(%arg0: !tt.ptr<f32> , %arg1: !tt.ptr<f32> ) attributes {noinline = false} {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    %3 = tt.load %2 : tensor<32x!tt.ptr<f32>>
    %4 = "tt.reduce"(%3) <{axis = 0 : i32}> ({
    ^bb0(%arg2: f32 , %arg3: f32 ):
      %9 = arith.addf %arg2, %arg3 : f32
      tt.reduce.return %9 : f32
    }) : (tensor<32xf32>) -> f32
    %5 = tt.splat %4 : f32 -> tensor<32xf32>
    %6 = arith.subf %3, %5 : tensor<32xf32>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>>
    %8 = tt.addptr %7, %0 : tensor<32x!tt.ptr<f32>>, tensor<32xi32>
    tt.store %8, %6 : tensor<32x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @fn1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f16>, [[PARAM_1_:%.+]]: !tt.ptr<f16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <f16> to tensor<32x!tt.ptr<f16>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tts.load"([[VAR_0_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<f16>>) -> tensor<32xf16>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_dot_000000_]] into [[VAR_2_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_1_]] : tensor<32xf16>) outs([[VAR_inserted_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_:.+]]: f16, [[init_:.+]]: f32) {
// CHECK:               [[VAR_13_:%.+]] = arith.extf [[in_]] : f16 to f32
// CHECK:               [[VAR_14_:%.+]] = arith.addf [[VAR_13_]], [[init_]] : f32
// CHECK:               linalg.yield [[VAR_14_]] : f32
// CHECK:             }
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.truncf [[VAR_extracted_]] : f32 to f16
// CHECK-DAG:       [[VAR_4_:%.+]] = tensor.empty() : tensor<32xf32>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_1_]] : tensor<32xf16>) outs([[VAR_4_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f16, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_13_1_:%.+]] = arith.extf [[IN_0_]] : f16 to f32
// CHECK:             linalg.yield [[VAR_13_1_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.extf [[VAR_3_]] : f16 to f32
// CHECK-DAG:       [[VAR_7_:%.+]] = tensor.empty() : tensor<32xf32>
// CHECK:           [[VAR_8_:%.+]] = linalg.fill ins([[VAR_6_]] : f32) outs([[VAR_7_]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_5_]], [[VAR_8_]] : tensor<32xf32>, tensor<32xf32>) outs([[VAR_5_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_2_:%.+]]: f32, [[IN_3_:%.+]]: f32, [[IN_4_:%.+]]: f32):
// CHECK:             [[VAR_13_2_:%.+]] = arith.divf [[IN_2_]], [[IN_3_]] : f32
// CHECK:             linalg.yield [[VAR_13_2_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <f16> to tensor<32x!tt.ptr<f16>>
// CHECK-DAG:       [[VAR_11_:%.+]] = tensor.empty() : tensor<32xf16>
// CHECK:           [[VAR_12_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_9_]] : tensor<32xf32>) outs([[VAR_11_]] : tensor<32xf16>) {
// CHECK:           ^bb0([[IN_5_:%.+]]: f32, [[IN_6_:%.+]]: f16):
// CHECK:             [[VAR_13_3_:%.+]] = arith.truncf [[IN_5_]] : f32 to f16
// CHECK:             linalg.yield [[VAR_13_3_]] : f16
// CHECK:           } -> tensor<32xf16>
// CHECK:           "tts.store"([[VAR_10_]], [[VAR_12_]]) <{static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<f16>>, tensor<32xf16>) -> ()
// CHECK:           return
// CHECK:         }
//
// CHECK-LABEL:  func.func @fn2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f16>, [[PARAM_1_:%.+]]: !tt.ptr<f16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <f16> to tensor<32x!tt.ptr<f16>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tts.load"([[VAR_0_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<f16>>) -> tensor<32xf16>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_dot_000000_]] into [[VAR_2_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_1_]] : tensor<32xf16>) outs([[VAR_inserted_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_:.+]]: f16, [[init_:.+]]: f32) {
// CHECK:               [[VAR_8_:%.+]] = arith.extf [[in_]] : f16 to f32
// CHECK:               [[VAR_9_:%.+]] = arith.addf [[VAR_8_]], [[init_]] : f32
// CHECK:               linalg.yield [[VAR_9_]] : f32
// CHECK:             }
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.truncf [[VAR_extracted_]] : f32 to f16
// CHECK-DAG:       [[VAR_4_:%.+]] = tensor.empty() : tensor<32xf16>
// CHECK:           [[VAR_5_:%.+]] = linalg.fill ins([[VAR_3_]] : f16) outs([[VAR_4_]] : tensor<32xf16>) -> tensor<32xf16>
// CHECK:           [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_1_]], [[VAR_5_]] : tensor<32xf16>, tensor<32xf16>) outs([[VAR_1_]] : tensor<32xf16>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f16, [[IN_1_:%.+]]: f16, [[IN_2_:%.+]]: f16):
// CHECK:             [[VAR_8_1_:%.+]] = arith.subf [[IN_0_]], [[IN_1_]] : f16
// CHECK:             linalg.yield [[VAR_8_1_]] : f16
// CHECK:           } -> tensor<32xf16>
// CHECK:           [[VAR_7_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <f16> to tensor<32x!tt.ptr<f16>>
// CHECK:           "tts.store"([[VAR_7_]], [[VAR_6_]]) <{static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<f16>>, tensor<32xf16>) -> ()
// CHECK:           return
// CHECK:         }
//
// CHECK-LABEL:  func.func @fn3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <bf16> to tensor<32x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tts.load"([[VAR_0_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<bf16>>) -> tensor<32xbf16>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_dot_000000_]] into [[VAR_2_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_1_]] : tensor<32xbf16>) outs([[VAR_inserted_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_:.+]]: bf16, [[init_:.+]]: f32) {
// CHECK:               [[VAR_8_:%.+]] = arith.extf [[in_]] : bf16 to f32
// CHECK:               [[VAR_9_:%.+]] = arith.addf [[VAR_8_]], [[init_]] : f32
// CHECK:               linalg.yield [[VAR_9_]] : f32
// CHECK:             }
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.truncf [[VAR_extracted_]] : f32 to bf16
// CHECK-DAG:       [[VAR_4_:%.+]] = tensor.empty() : tensor<32xbf16>
// CHECK:           [[VAR_5_:%.+]] = linalg.fill ins([[VAR_3_]] : bf16) outs([[VAR_4_]] : tensor<32xbf16>) -> tensor<32xbf16>
// CHECK:           [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_1_]], [[VAR_5_]] : tensor<32xbf16>, tensor<32xbf16>) outs([[VAR_1_]] : tensor<32xbf16>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: bf16, [[IN_1_:%.+]]: bf16, [[IN_2_:%.+]]: bf16):
// CHECK:             [[VAR_8_1_:%.+]] = arith.subf [[IN_0_]], [[IN_1_]] : bf16
// CHECK:             linalg.yield [[VAR_8_1_]] : bf16
// CHECK:           } -> tensor<32xbf16>
// CHECK:           [[VAR_7_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <bf16> to tensor<32x!tt.ptr<bf16>>
// CHECK:           "tts.store"([[VAR_7_]], [[VAR_6_]]) <{static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<bf16>>, tensor<32xbf16>) -> ()
// CHECK:           return
// CHECK:         }
//
// CHECK-LABEL:  func.func @fn4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<32x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tts.load"([[VAR_0_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<f32>>) -> tensor<32xf32>
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.alloc_tensor() : tensor<f32>
// CHECK:           [[VAR_inserted_:%.+]] = tensor.insert [[CST_0_dot_000000_]] into [[VAR_2_]][] : tensor<f32>
// CHECK:           [[VAR_reduced_:%.+]] = linalg.reduce ins([[VAR_1_]] : tensor<32xf32>) outs([[VAR_inserted_]] : tensor<f32>) dimensions = [0]
// CHECK:             ([[in_:.+]]: f32, [[init_:.+]]: f32) {
// CHECK:               [[VAR_7_:%.+]] = arith.addf [[in_]], [[init_]] : f32
// CHECK:               linalg.yield [[VAR_7_]] : f32
// CHECK:             }
// CHECK-DAG:       [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]][] : tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tensor.empty() : tensor<32xf32>
// CHECK:           [[VAR_4_:%.+]] = linalg.fill ins([[VAR_extracted_]] : f32) outs([[VAR_3_]] : tensor<32xf32>) -> tensor<32xf32>
// CHECK:           [[VAR_5_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_1_]], [[VAR_4_]] : tensor<32xf32>, tensor<32xf32>) outs([[VAR_1_]] : tensor<32xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32, [[IN_2_:%.+]]: f32):
// CHECK:             [[VAR_7_1_:%.+]] = arith.subf [[IN_0_]], [[IN_1_]] : f32
// CHECK:             linalg.yield [[VAR_7_1_]] : f32
// CHECK:           } -> tensor<32xf32>
// CHECK:           [[VAR_6_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [32], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<32x!tt.ptr<f32>>
// CHECK:           "tts.store"([[VAR_6_]], [[VAR_5_]]) <{static_mask_dims = array<i64>}> : (tensor<32x!tt.ptr<f32>>, tensor<32xf32>) -> ()
// CHECK:           return
// CHECK:         }
