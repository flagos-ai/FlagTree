// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

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
// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]]:2 = scf.for [[VAR_arg5_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg6_:%.+]] = [[VAR_cst_]], [[VAR_arg7_:%.+]] = [[VAR_2_]]) -> (tensor<128x128xf32>, index) {
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addi [[VAR_arg7_]], [[CST_128_]] : index
// CHECK:             [[VAR_9_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [128, 128], strides: [1, 1], offsets: {{.}}[[VAR_8_]], 0], shape: [0, 0], order: [] : <f32> to tensor<128x128x!tt.ptr<f32>>
// CHECK:             [[VAR_10_:%.+]] = "tts.load"([[VAR_9_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32>>) -> tensor<128x128xf32>
// CHECK:             [[VAR_11_:%.+]] = math.exp [[VAR_10_]] : tensor<128x128xf32>
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.addf [[VAR_arg6_]], [[VAR_11_]] : tensor<128x128xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addi [[VAR_arg7_]], [[VAR_arg5_]] : index
// CHECK:             scf.yield [[VAR_12_]], [[VAR_13_]] : tensor<128x128xf32>, index
// CHECK:           }
// CHECK:           [[VAR_4_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK:           [[VAR_6_:%.+]] = arith.addi [[VAR_5_]], [[CST_128_]] : index
// CHECK:           [[VAR_7_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [128, 128], strides: [1, 1], offsets: {{.}}[[VAR_6_]], 0], shape: [0, 0], order: [] : <f32> to tensor<128x128x!tt.ptr<f32>>
// CHECK:           "tts.store"([[VAR_7_]], [[VAR_3_]]#0) <{static_mask_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32>>, tensor<128x128xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
