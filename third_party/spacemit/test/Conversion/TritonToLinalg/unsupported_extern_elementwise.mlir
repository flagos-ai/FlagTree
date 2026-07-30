// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s

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
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xi32>, [[PARAM_1_:%.+]]: memref<*xi32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = tensor.empty() : tensor<8xi32>
// CHECK:           [[VAR_1_:%.+]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs([[VAR_0_]] : tensor<8xi32>) {
// CHECK:           ^bb0([[out:.+]]: i32):
// CHECK:             [[VAR_4_:%.+]] = linalg.index 0 : index
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : index to i32
// CHECK:             linalg.yield [[VAR_5_]] : i32
// CHECK:           } -> tensor<8xi32>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [8], strides: [1] : memref<*xi32> to memref<8xi32, strided<[1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<8xi32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<8xi32, strided<[1]>> to memref<8xi32>
// CHECK:           [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<8xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tt.extern_elementwise [[VAR_2_]], [[VAR_1_]] {libname = "", libpath = "", pure = true, symbol = "some_symbol"} : (tensor<8xi32>, tensor<8xi32>) -> tensor<8xi32>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [8], strides: [1] : memref<*xi32> to memref<8xi32, strided<[1]>>
// CHECK:           bufferization.materialize_in_destination [[VAR_3_]] in writable [[VAR_reinterpret_cast_0_]]
// CHECK:           return
// CHECK:         }
