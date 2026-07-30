// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %cf0 = arith.constant 0.000000e+00 : f32
    %tensor_cf0 = tt.splat %cf0 : f32 -> tensor<128x128xf32>
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %sum_out, %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%sum_iter = %tensor_cf0,  %ptr_iter = %2) ->  (tensor<128x128xf32>, !tt.ptr<f32> ) {
      %3 = tt.splat %ptr_iter : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
      // source = %arg1, offset = [%1, 0], size = [128, 128], strides = [0, 0]
      %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
      %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
      %6 = tt.broadcast %5 : tensor<1x128xi32> -> tensor<128x128xi32>
      // offset = [0, 0], size = [128, 128], strides = [0, 1]
      %7 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32>
      %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %9 = tt.broadcast %8 : tensor<128x1xi32> -> tensor<128x128xi32>
      // offset = [128, 0], size = [128, 128], strides = [1, 0]
      %10 = arith.addi %6, %9 : tensor<128x128xi32>
      // offset = [128, 0], size = [128, 128], strides = [1, 1]
      %11 = tt.addptr %3, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      // source = %arg1, offset = [%1 + 128, 0], size = [128, 128], strides = [1, 1]
      %12 = tt.load %11 : tensor<128x128x!tt.ptr<f32>>
      %17 = math.exp %12 : tensor<128x128xf32>
      %sum_next = arith.addf %sum_iter, %17 : tensor<128x128xf32>
      %cast_i = arith.index_cast %i : index to i32
      %ptr_next = tt.addptr %ptr_iter, %cast_i : !tt.ptr<f32>, i32
      // source = %arg1, offset = %1 + %i, size = 1, strides = 0
      scf.yield %sum_next, %ptr_next : tensor<128x128xf32>, !tt.ptr<f32>
    }
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %6 = tt.broadcast %5 : tensor<1x128xi32> -> tensor<128x128xi32>
    // offset = [0, 0], size = [128, 128], strides = [0, 1]
    %7 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32>
    %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %9 = tt.broadcast %8 : tensor<128x1xi32> -> tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [1, 0]
    %10 = arith.addi %6, %9 : tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [1, 1]
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = %18, size = 1, strides = 0
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    // source = arg0, offset = [%18, 0], size = [128, 128], strides = [0, 0]
    %21 = tt.addptr %20, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    // source = %arg0, offset = [%18 + 128, 0], size = [128, 128], strides = [1, 1]
    tt.store %21, %sum_out : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<128x128xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : f32) outs([[VAR_0_]] : tensor<128x128xf32>) -> tensor<128x128xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[PARAM_8_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[VAR_2_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]]:2 = scf.for [[VAR_arg11_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg12_:%.+]] = [[VAR_1_]], [[VAR_arg14_:%.+]] = [[VAR_3_]]) -> (tensor<128x128xf32>, index) {
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addi [[VAR_arg14_]], [[CST_128_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_8_]]{{.}}, sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<128x128xf32>
// CHECK:             memref.copy [[VAR_reinterpret_cast_0_]], [[RES_]] : memref<128x128xf32, strided<[1, 1], offset: ?>> to memref<128x128xf32>
// CHECK:             [[VAR_9_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128x128xf32>
// CHECK:             [[VAR_10_:%.+]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_9_]] : tensor<128x128xf32>) outs([[VAR_9_]] : tensor<128x128xf32>) {
// CHECK:             ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:               [[VAR_15_:%.+]] = math.exp [[IN_0_]] : f32
// CHECK:               linalg.yield [[VAR_15_]] : f32
// CHECK:             } -> tensor<128x128xf32>
// CHECK:             [[VAR_11_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_arg12_]], [[VAR_10_]] : tensor<128x128xf32>, tensor<128x128xf32>) outs([[VAR_arg12_]] : tensor<128x128xf32>) {
// CHECK:             ^bb0([[IN_2_:%.+]]: f32, [[IN_3_:%.+]]: f32, [[IN_4_:%.+]]: f32):
// CHECK:               [[VAR_15_1_:%.+]] = arith.addf [[IN_2_]], [[IN_3_]] : f32
// CHECK:               linalg.yield [[VAR_15_1_]] : f32
// CHECK:             } -> tensor<128x128xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addi [[VAR_arg14_]], [[VAR_arg11_]] : index
// CHECK:             scf.yield [[VAR_11_]], [[VAR_13_]] : tensor<128x128xf32>, index
// CHECK:           }
// CHECK:           [[VAR_5_:%.+]] = arith.muli [[PARAM_8_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_5_]] : i32 to index
// CHECK:           [[VAR_7_:%.+]] = arith.addi [[VAR_6_]], [[CST_128_]] : index
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_7_]]{{.}}, sizes: [128, 128], strides: [1, 1] : memref<*xf32> to memref<128x128xf32, strided<[1, 1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_4_]]#0 in writable [[VAR_reinterpret_cast_]] : (tensor<128x128xf32>, memref<128x128xf32, strided<[1, 1], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
