// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @nested_use_same_level_loop_result(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %2 = tt.splat %arg2 : i32 -> tensor<2x1xi32>
    %3 = arith.muli %1, %2 : tensor<2x1xi32>
    %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1x2xi32>
    %6 = arith.muli %4, %5 : tensor<1x2xi32>
    %7 = tt.broadcast %3 : tensor<2x1xi32> -> tensor<2x2xi32>
    %8 = tt.broadcast %6 : tensor<1x2xi32> -> tensor<2x2xi32>
    %9 = arith.addi %7, %8 : tensor<2x2xi32>
    %10 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x2x!tt.ptr<f32>>
    %11 = tt.addptr %10, %9 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x1x!tt.ptr<f32>>
    %13 = tt.addptr %12, %3 : tensor<2x1x!tt.ptr<f32>>, tensor<2x1xi32>
    %14 = tt.broadcast %13 : tensor<2x1x!tt.ptr<f32>> -> tensor<2x2x!tt.ptr<f32>>
    %15 = tt.addptr %14, %8 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
    %16 = arith.muli %arg3, %c2_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18 = arith.muli %arg3, %c2_i32 : i32
    %19 = tt.splat %18 : i32 -> tensor<2x2xi32>
    %20 = arith.muli %arg3, %c2_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<2x2xi32>
    %22:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %23 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5) -> (tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %26 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %26 : tensor<2x2x!tt.ptr<f32>>
      }
      %24:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %23, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        %27 = tt.addptr %arg8, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %28 = tt.load %27 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %26 : tensor<2x2x!tt.ptr<f32>>
        %29 = tt.addptr %arg9, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %30 = tt.addptr %29, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        tt.store %30, %28 : tensor<2x2x!tt.ptr<f32>>
        %31 = tt.addptr %30, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %32 = tt.addptr %27, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %32, %31 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %25 = tt.addptr %24#0, %21 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %25, %24#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK:         tt.func public @nested_use_same_level_loop_result([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = tt.expand_dims [[VAR_0_]] {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tt.splat [[PARAM_2_]] : i32 -> tensor<2x1xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[VAR_2_]] : tensor<2x1xi32>
// CHECK-DAG:       [[VAR_4_:%.+]] = tt.expand_dims [[VAR_0_]] {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tt.splat [[PARAM_3_]] : i32 -> tensor<1x2xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.muli [[VAR_4_]], [[VAR_5_]] : tensor<1x2xi32>
// CHECK-DAG:       [[VAR_7_:%.+]] = tt.broadcast [[VAR_3_]] : tensor<2x1xi32> -> tensor<2x2xi32>
// CHECK:           [[VAR_8_:%.+]] = tt.broadcast [[VAR_6_]] : tensor<1x2xi32> -> tensor<2x2xi32>
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.addi [[VAR_7_]], [[VAR_8_]] : tensor<2x2xi32>
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.muli [[PARAM_3_]], [[CST_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = tt.splat [[VAR_10_]] : i32 -> tensor<2x2xi32>
// CHECK-DAG:       [[VAR_12_:%.+]]:2 = scf.for [[VAR_arg4_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg5_:%.+]] = [[VAR_9_]], [[VAR_arg6_:%.+]] = [[VAR_9_]]) -> (tensor<2x2xi32>, tensor<2x2xi32>)  : i32 {
// CHECK-DAG:         [[VAR_13_:%.+]] = scf.for [[VAR_arg7_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg8_:%.+]] = [[VAR_arg5_]]) -> (tensor<2x2xi32>)  : i32 {
// CHECK:               [[VAR_16_:%.+]] = arith.addi [[VAR_arg8_]], [[VAR_11_]] : tensor<2x2xi32>
// CHECK:               scf.yield [[VAR_16_]] : tensor<2x2xi32>
// CHECK:             }
// CHECK-DAG:         [[VAR_14_:%.+]]:2 = scf.for [[VAR_arg7_1_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg8_1_:%.+]] = [[VAR_13_]], [[VAR_arg9_:%.+]] = [[VAR_arg6_]]) -> (tensor<2x2xi32>, tensor<2x2xi32>)  : i32 {
// CHECK-DAG:           [[VAR_16_1_:%.+]] = tts.gather [[PARAM_0_]]{{.}}[[VAR_arg8_1_]]{{.}} : (<f32>, tensor<2x2xi32>) -> tensor<2x2xf32>
// CHECK-DAG:           [[VAR_17_:%.+]] = arith.addi [[VAR_arg8_1_]], [[VAR_11_]] : tensor<2x2xi32>
// CHECK:               [[VAR_18_:%.+]] = tts.gather [[PARAM_0_]]{{.}}[[VAR_17_]]{{.}} : (<f32>, tensor<2x2xi32>) -> tensor<2x2xf32>
// CHECK:               tts.scatter [[VAR_16_1_]] into [[PARAM_1_]]{{.}}[[VAR_arg9_]]{{.}} : tensor<2x2xf32> into (<f32>, tensor<2x2xi32>)
// CHECK:               [[VAR_19_:%.+]] = arith.addi [[VAR_arg9_]], [[VAR_11_]] : tensor<2x2xi32>
// CHECK:               [[VAR_20_:%.+]] = arith.addi [[VAR_19_]], [[VAR_11_]] : tensor<2x2xi32>
// CHECK:               tts.scatter [[VAR_18_]] into [[PARAM_1_]]{{.}}[[VAR_20_]]{{.}} : tensor<2x2xf32> into (<f32>, tensor<2x2xi32>)
// CHECK-DAG:           [[VAR_21_:%.+]] = arith.addi [[VAR_20_]], [[VAR_11_]] : tensor<2x2xi32>
// CHECK-DAG:           [[VAR_22_:%.+]] = arith.addi [[VAR_17_]], [[VAR_11_]] : tensor<2x2xi32>
// CHECK:               scf.yield [[VAR_22_]], [[VAR_21_]] : tensor<2x2xi32>, tensor<2x2xi32>
// CHECK:             }
// CHECK:             [[VAR_15_:%.+]] = arith.addi [[VAR_14_]]#0, [[VAR_11_]] : tensor<2x2xi32>
// CHECK:             scf.yield [[VAR_15_]], [[VAR_14_]]#1 : tensor<2x2xi32>, tensor<2x2xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
