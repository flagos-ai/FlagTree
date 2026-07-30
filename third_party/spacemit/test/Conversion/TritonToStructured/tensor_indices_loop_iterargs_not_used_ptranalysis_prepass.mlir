// IR obtained from "test_integer_tensor" in python/examples/test_tensor_index_iterargs.py

// RUN: triton-shared-opt --triton-to-structured="run-prepass-only" %s | FileCheck %s
module {
  tt.func public @test_1(%arg0: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<4> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %2:2 = scf.for %arg1 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg2 = %0, %arg3 = %0) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %3 = tt.addptr %1, %arg2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %4 = arith.sitofp %arg3 : tensor<4xi32> to tensor<4xf32>
      tt.store %3, %4 : tensor<4x!tt.ptr<f32>>
      %5 = arith.addi %arg2, %cst : tensor<4xi32>
      %6 = arith.addi %arg3, %cst : tensor<4xi32>
      scf.yield %5, %6 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

// CHECK:         tt.func public @test_1([[arg0_:.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<4> : tensor<4xi32>
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tt.splat [[arg0_]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK-DAG:       [[structured_:%.+]], [[offsets_:%.+]], [[VAR_strides_:%.+]] = "tts.get_structured_state"([[VAR_0_]]) <{resultSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<4xi32>) -> (tensor<4xi32>, index, index)
// CHECK-DAG:       [[VAR_2_:%.+]]:6 = scf.for [[VAR_arg1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg2_:%.+]] = [[structured_]], [[VAR_arg3_:%.+]] = [[offsets_]], [[VAR_arg4_:%.+]] = [[VAR_strides_]], [[VAR_arg5_:%.+]] = [[structured_]], [[VAR_arg6_:%.+]] = [[offsets_]], [[VAR_arg7_:%.+]] = [[VAR_strides_]]) -> (tensor<4xi32>, index, index, tensor<4xi32>, index, index)  : i32 {
// CHECK-DAG:         [[VAR_3_:%.+]] = tt.addptr [[VAR_1_]], [[VAR_arg2_]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.sitofp [[VAR_arg5_]] : tensor<4xi32> to tensor<4xf32>
// CHECK:             tt.store [[VAR_3_]], [[VAR_4_]] : tensor<4x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_arg2_]], [[VAR_cst_]] : tensor<4xi32>
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_arg5_]], [[VAR_cst_]] : tensor<4xi32>
// CHECK-DAG:         [[structured_3_:%.+]], [[offsets_4_:%.+]], [[VAR_strides_5_:%.+]] = "tts.get_structured_state"([[VAR_5_]]) <{resultSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<4xi32>) -> (tensor<4xi32>, index, index)
// CHECK-DAG:         [[structured_6_:%.+]], [[offsets_7_:%.+]], [[VAR_strides_8_:%.+]] = "tts.get_structured_state"([[VAR_6_]]) <{resultSegmentSizes = array<i32: 1, 1, 1>}> : (tensor<4xi32>) -> (tensor<4xi32>, index, index)
// CHECK:             scf.yield [[structured_3_]], [[offsets_4_]], [[VAR_strides_5_]], [[structured_6_]], [[offsets_7_]], [[VAR_strides_8_]] : tensor<4xi32>, index, index, tensor<4xi32>, index, index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
