// RUN: triton-shared-opt --triton-to-linalg --split-input-file %s | FileCheck %s

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

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @argmax_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xi32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[PARAM_2_]] : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_1_]] : tensor<4096xi32>) {
// CHECK:           ^bb0([[out_:.+]]: i32):
// CHECK:             [[VAR_10_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i32
// CHECK:             linalg.yield [[VAR_11_]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [4096], strides: [1] : memref<*xf32> to memref<4096xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<4096xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4096xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tensor.empty() : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = linalg.fill ins([[CST_0_]] : f32) outs([[VAR_5_]] : tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tensor.empty() : tensor<i32>
// CHECK:           [[VAR_8_:%.+]] = linalg.fill ins([[CST_minus_1_]] : i32) outs([[VAR_7_]] : tensor<i32>) -> tensor<i32>
// CHECK:           [[VAR_reduced_:%.+]]:2 = linalg.reduce ins([[VAR_4_]], [[VAR_2_]] : tensor<4096xf32>, tensor<4096xi32>) outs([[VAR_6_]], [[VAR_8_]] : tensor<f32>, tensor<i32>) dimensions = [0]
// CHECK:             ([[in:.+]]: f32, [[in_1:.+]]: i32, [[init:.+]]: f32, [[init_2:.+]]: i32) {
// CHECK-DAG:           [[VAR_10_1_:%.+]] = arith.cmpf oeq, [[in]], [[init]] : f32
// CHECK-DAG:           [[VAR_11_1_:%.+]] = arith.cmpi slt, [[in_1]], [[init_2]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_12_:%.+]] = arith.andi [[VAR_10_1_]], [[VAR_11_1_]] : i1
// CHECK-DAG:           [[VAR_13_:%.+]] = arith.cmpf ogt, [[in]], [[init]] : f32
// CHECK:               [[VAR_14_:%.+]] = arith.ori [[VAR_13_]], [[VAR_12_]] : i1
// CHECK-DAG:           [[VAR_15_:%.+]] = arith.select [[VAR_14_]], [[in]], [[init]] : f32
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.select [[VAR_14_]], [[in_1]], [[init_2]] : i32
// CHECK:               linalg.yield [[VAR_15_]], [[VAR_16_]] : f32, i32
// CHECK:             }
// CHECK-DAG:       [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]]#1[] : tensor<i32>
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK:           [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_9_]]{{.}}, sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:           affine.store [[VAR_extracted_]], [[VAR_reinterpret_cast_0_]][0] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }

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
// CHECK-LABEL:  func.func @argmin_012
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xi32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[PARAM_6_]], [[PARAM_2_]] : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = tensor.empty() : tensor<4096xi32>
// CHECK:           [[VAR_2_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_1_]] : tensor<4096xi32>) {
// CHECK:           ^bb0([[out_:.+]]: i32):
// CHECK:             [[VAR_10_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i32
// CHECK:             linalg.yield [[VAR_11_]] : i32
// CHECK:           } -> tensor<4096xi32>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [4096], strides: [1] : memref<*xf32> to memref<4096xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<4096xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<4096xf32, strided<[1], offset: ?>> to memref<4096xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4096xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tensor.empty() : tensor<f32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = linalg.fill ins([[CST_0_]] : f32) outs([[VAR_5_]] : tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tensor.empty() : tensor<i32>
// CHECK:           [[VAR_8_:%.+]] = linalg.fill ins([[CST_minus_1_]] : i32) outs([[VAR_7_]] : tensor<i32>) -> tensor<i32>
// CHECK:           [[VAR_reduced_:%.+]]:2 = linalg.reduce ins([[VAR_4_]], [[VAR_2_]] : tensor<4096xf32>, tensor<4096xi32>) outs([[VAR_6_]], [[VAR_8_]] : tensor<f32>, tensor<i32>) dimensions = [0]
// CHECK:             ([[in:.+]]: f32, [[in_1:.+]]: i32, [[init:.+]]: f32, [[init_2:.+]]: i32) {
// CHECK-DAG:           [[VAR_10_1_:%.+]] = arith.cmpf oeq, [[in]], [[init]] : f32
// CHECK-DAG:           [[VAR_11_1_:%.+]] = arith.cmpi slt, [[in_1]], [[init_2]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_12_:%.+]] = arith.andi [[VAR_10_1_]], [[VAR_11_1_]] : i1
// CHECK-DAG:           [[VAR_13_:%.+]] = arith.cmpf olt, [[in]], [[init]] : f32
// CHECK:               [[VAR_14_:%.+]] = arith.ori [[VAR_13_]], [[VAR_12_]] : i1
// CHECK-DAG:           [[VAR_15_:%.+]] = arith.select [[VAR_14_]], [[in]], [[init]] : f32
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.select [[VAR_14_]], [[in_1]], [[init_2]] : i32
// CHECK:               linalg.yield [[VAR_15_]], [[VAR_16_]] : f32, i32
// CHECK:             }
// CHECK-DAG:       [[VAR_extracted_:%.+]] = tensor.extract [[VAR_reduced_]]#1[] : tensor<i32>
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK:           [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_9_]]{{.}}, sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:           affine.store [[VAR_extracted_]], [[VAR_reinterpret_cast_0_]][0] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }
