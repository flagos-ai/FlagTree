// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    // source = %arg1, offset = [%1, 0], size = [128, 128], strides = [0, 0]
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %6 = tt.broadcast %5 : tensor<1x128xi32> -> tensor<128x128xi32>
    // offset = [0, 0], size = [128, 128], strides = [0, 1]
    %7 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32>
    // offset = 128, size = 128, strides = 1
    %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %9 = tt.broadcast %8 : tensor<128x1xi32> -> tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [1, 0]
    %10 = arith.addi %6, %9 : tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [1, 1]
    %11 = tt.addptr %3, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    // source = %arg1, offset = [%1 + 128, 0], size = [128, 128], strides = [1, 1]
    %12 = tt.load %11 : tensor<128x128x!tt.ptr<f32>>
    %17 = math.exp %12 : tensor<128x128xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = %18, size = 1, strides = 0
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    // source = arg0, offset = [%18, 0], size = [128, 128], strides = [0, 0]
    %21 = tt.addptr %20, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    // source = %arg0, offset = [%18 + 128, 0], size = [128, 128], strides = [1, 1]
    tt.store %21, %17 : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32) {
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[PARAM_8_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK:           [[VAR_2_:%.+]] = arith.addi [[VAR_1_]], [[CST_128_]] : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_2_]]{{.}}, sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<128x128xf32>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<128x128xf32, strided<[1, 1], offset: ?>> to memref<128x128xf32>
// CHECK:           [[VAR_3_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128x128xf32>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_3_]] : tensor<128x128xf32>) outs([[VAR_3_]] : tensor<128x128xf32>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_8_:%.+]] = math.exp [[IN_0_]] : f32
// CHECK:             linalg.yield [[VAR_8_]] : f32
// CHECK:           } -> tensor<128x128xf32>
// CHECK:           [[VAR_5_:%.+]] = arith.muli [[PARAM_8_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_5_]] : i32 to index
// CHECK:           [[VAR_7_:%.+]] = arith.addi [[VAR_6_]], [[CST_128_]] : index
// CHECK:           [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_7_]]{{.}}, sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_4_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<128x128xf32>, memref<128x128xf32, strided<[1, 1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
