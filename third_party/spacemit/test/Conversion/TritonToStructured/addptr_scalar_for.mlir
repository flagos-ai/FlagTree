// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // source = %arg1, offset = %1, size = 1, strides = 0
    %cf0 = arith.constant 0.000000e+00 : f32
    %tensor_cf0 = tt.splat %cf0 : f32 -> tensor<1024xf32>
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %_ptr, %sum_out = scf.for %i = %c0 to %c12 step %c3 iter_args(%ptr_iter = %2, %sum_iter = %tensor_cf0) ->  (!tt.ptr<f32>, tensor<1024xf32>) {
      %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
      // offset = 0, size = 1024, strides = 1
      %4 = tt.splat %ptr_iter : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
      // source = %arg1, offset = %1, size = 1024, strides = 0
      %5 = tt.addptr %4, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      // source = %arg1, offset = %1, size = 1024, strides = 1
      %8 = tt.load %5 : tensor<1024x!tt.ptr<f32>>
      %9 = math.exp %8 : tensor<1024xf32>
      %sum_next = arith.addf %sum_iter, %9 : tensor<1024xf32>
      %cast_i = arith.index_cast %i : index to i32
      %ptr_next = tt.addptr %ptr_iter, %cast_i : !tt.ptr<f32>, i32
      // source = %arg1, offset = %1 + %i, size = 1, strides = 0
      scf.yield %ptr_next, %sum_next : !tt.ptr<f32>, tensor<1024xf32>
    }
    %10 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %21 = tt.addptr %20, %10 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %21, %sum_out : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : tensor<1024xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]]:2 = scf.for [[VAR_arg5_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg7_:%.+]] = [[VAR_2_]], [[VAR_arg6_:%.+]] = [[VAR_cst_]]) -> (index, tensor<1024xf32>) {
// CHECK-DAG:         [[VAR_7_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [1024], strides: [1], offsets: {{.}}[[VAR_arg7_]]{{.}}, shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
// CHECK:             [[VAR_8_:%.+]] = "tts.load"([[VAR_7_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
// CHECK:             [[VAR_9_:%.+]] = math.exp [[VAR_8_]] : tensor<1024xf32>
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.addf [[VAR_arg6_]], [[VAR_9_]] : tensor<1024xf32>
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addi [[VAR_arg7_]], [[VAR_arg5_]] : index
// CHECK:             scf.yield [[VAR_11_]], [[VAR_10_]] : index, tensor<1024xf32>
// CHECK:           }
// CHECK:           [[VAR_4_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK:           [[VAR_6_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [1024], strides: [1], offsets: {{.}}[[VAR_5_]]{{.}}, shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
// CHECK:           "tts.store"([[VAR_6_]], [[VAR_3_]]#1) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
