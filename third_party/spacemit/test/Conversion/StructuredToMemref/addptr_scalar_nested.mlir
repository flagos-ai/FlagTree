// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1, size = 1, strides = 0
    %3 = arith.muli %0, %arg3 : i32
    %4 = tt.addptr %2, %3 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1+%3, size = 1, strides = 0
    %5 = arith.muli %0, %arg4 : i32
    %6 = tt.addptr %4, %5 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1+%3+%5, size = 1, strides = 0
    %7 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // offset = 0, size = 1024, strides = 1
    %8 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // source = arg1, offset = %1, size = 1024, strides = 0
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // source = arg1, offset = %1+%3+%5, size = 1024, strides = 1
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    %17 = math.exp %10 : tensor<1024xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = %18, size = 1, strides = 0
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // source = arg0, offset = %18, size = 1024, strides = 0
    %21 = tt.addptr %20, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // source = arg0, offset = %18, size = 1024, strides = 1
    tt.store %21, %17 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[PARAM_8_]], [[PARAM_2_]] : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[PARAM_8_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.addi [[VAR_1_]], [[VAR_3_]] : index
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.muli [[PARAM_8_]], [[PARAM_4_]] : i32
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_5_]] : i32 to index
// CHECK:           [[VAR_7_:%.+]] = arith.addi [[VAR_4_]], [[VAR_6_]] : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_7_]]{{.}}, sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1024xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32>
// CHECK:           [[VAR_8_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1024xf32>
// CHECK:           [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins([[VAR_8_]] : tensor<1024xf32>) outs([[VAR_8_]] : tensor<1024xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_10_:%.+]] = math.exp [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_10_]] : f32
// CHECK:           } -> tensor<1024xf32>
// CHECK:           [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_9_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<1024xf32>, memref<1024xf32, strided<[1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
