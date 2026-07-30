// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func public @unsplat_kernel(%arg0: !tt.ptr<i32> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<42> : tensor<1xi32>
    %c42_i32 = arith.constant 42 : i32
    %0 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %1 = tt.load %0 : tensor<1x!tt.ptr<i32>>
    %2 = arith.cmpi sgt, %1, %cst : tensor<1xi32>
    %3 = "tt.reduce"(%2) <{axis = 0 : i32}> ({
    ^bb0(%arg1: i1, %arg2: i1):
      tt.reduce.return %arg1 : i1
    }) : (tensor<1xi1>) -> i1
    scf.if %3 {
      tt.store %arg0, %c42_i32 : !tt.ptr<i32>
    }
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @unsplat_kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       [[CST_42_:%.+]] = arith.constant 42 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<1xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_42_]] : i32) outs([[VAR_0_]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = linalg.fill ins([[CST_0_]] : i32) outs([[VAR_0_]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK-DAG:       [[VAR_cast_:%.+]] = memref.cast [[PARAM_0_]] : memref<*xi32> to memref<?xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = bufferization.to_tensor [[VAR_cast_]] restrict : memref<?xi32> to tensor<?xi32>
// CHECK-DAG:       [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_2_]] : tensor<1xi32>) outs([[VAR_0_]] : tensor<1xi32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: i32, [[IN_1_:%.+]]: i32):
// CHECK:             [[VAR_7_:%.+]] = arith.index_cast [[IN_0_]] : i32 to index
// CHECK:             [[VAR_extracted_0_:%.+]] = tensor.extract [[VAR_3_]]{{.}}[[VAR_7_]]{{.}} : tensor<?xi32>
// CHECK:             linalg.yield [[VAR_extracted_0_]] : i32
// CHECK:           } -> tensor<1xi32>
// CHECK:           [[VAR_5_:%.+]] = tensor.empty() : tensor<1xi1>
// CHECK:           [[VAR_6_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins([[VAR_4_]], [[VAR_1_]] : tensor<1xi32>, tensor<1xi32>) outs([[VAR_5_]] : tensor<1xi1>) {
// CHECK:           ^bb0([[IN_2_:%.+]]: i32, [[IN_3_:%.+]]: i32, [[IN_4_:%.+]]: i1):
// CHECK:             [[VAR_7_1_:%.+]] = arith.cmpi sgt, [[IN_2_]], [[IN_3_]] : i32
// CHECK:             linalg.yield [[VAR_7_1_]] : i1
// CHECK:           } -> tensor<1xi1>
// CHECK:           [[VAR_extracted_:%.+]] = tensor.extract [[VAR_6_]]{{.}}[[CST_0_1_]]{{.}} : tensor<1xi1>
// CHECK:           scf.if [[VAR_extracted_]] {
// CHECK:             [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[CST_0_1_]]{{.}}, sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:             affine.store [[CST_42_]], [[VAR_reinterpret_cast_]][0] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           return
// CHECK:         }
