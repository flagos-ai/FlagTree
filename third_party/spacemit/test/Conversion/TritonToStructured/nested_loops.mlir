// RUN: triton-shared-opt --split-input-file --triton-to-structured --canonicalize --remove-dead-values %s | FileCheck %s

module {
  tt.func public @nested2_complex_body(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<3> : tensor<2x2xi32>
    %cst_0 = arith.constant dense<1> : tensor<2x2xi32>
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
    %16 = arith.muli %arg2, %c2_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %19 = tt.addptr %arg5, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %20 = tt.addptr %arg6, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %21:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %19, %arg9 = %20) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %26 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %26 : tensor<2x2x!tt.ptr<f32>>
        %27 = tt.addptr %arg8, %cst : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %28 = tt.addptr %arg9, %cst : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %27, %28 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %22 = tt.addptr %arg5, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %23 = tt.addptr %22, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %24 = tt.addptr %arg6, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %25 = tt.addptr %24, %cst_0 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %23, %25 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
  tt.func public @nested2_use_loop_results(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c4_i32 = arith.constant 4 : i32
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
    %16 = arith.muli %arg3, %c4_i32 : i32
    %17 = tt.splat %16 : i32 -> tensor<2x2xi32>
    %18:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %19 = tt.load %arg5 : tensor<2x2x!tt.ptr<f32>>
      tt.store %arg6, %19 : tensor<2x2x!tt.ptr<f32>>
      %20 = tt.addptr %arg5, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %21 = tt.addptr %arg6, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      %22:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %20, %arg9 = %21) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %23 = tt.load %arg8 : tensor<2x2x!tt.ptr<f32>>
        tt.store %arg9, %23 : tensor<2x2x!tt.ptr<f32>>
        %24 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %25 = tt.addptr %arg9, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        scf.yield %24, %25 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      scf.yield %22#0, %22#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
  tt.func public @nested3(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32) attributes {noinline = false} {
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
    %20:2 = scf.for %arg4 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg5 = %11, %arg6 = %15) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
      %21 = tt.load %arg5 : tensor<2x2x!tt.ptr<f32>>
      %22:2 = scf.for %arg7 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg8 = %arg5, %arg9 = %arg6) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
        %24 = tt.addptr %arg8, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
        %25 = tt.load %24 : tensor<2x2x!tt.ptr<f32>>
        %26:2 = scf.for %arg10 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg11 = %24, %arg12 = %arg9) -> (tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>)  : i32 {
          %27 = tt.addptr %arg11, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          %28 = tt.load %27 : tensor<2x2x!tt.ptr<f32>>
          tt.store %arg12, %21 : tensor<2x2x!tt.ptr<f32>>
          %29 = tt.addptr %arg12, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %29, %25 : tensor<2x2x!tt.ptr<f32>>
          %30 = tt.addptr %29, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          tt.store %30, %28 : tensor<2x2x!tt.ptr<f32>>
          %31 = tt.addptr %30, %17 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
          scf.yield %27, %31 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
        }
        scf.yield %26#0, %26#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
      }
      %23 = tt.addptr %22#0, %19 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2xi32>
      scf.yield %23, %22#1 : tensor<2x2x!tt.ptr<f32>>, tensor<2x2x!tt.ptr<f32>>
    }
    tt.return
  }
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

// CHECK:         tt.func public @nested2_complex_body([[arg0_:.+]]: !tt.ptr<f32>, [[arg1_:.+]]: !tt.ptr<f32>, [[arg2_:.+]]: i32, [[arg3_:.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[arg2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[arg3_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[arg2_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[arg3_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.muli [[arg2_]], [[CST_2_]] : i32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[VAR_4_]] : i32 to index
// CHECK-DAG:       [[VAR_7_:%.+]]:2 = scf.for [[VAR_arg4_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_1_]] iter_args([[VAR_arg5_:%.+]] = [[CST_0_]], [[VAR_arg6_:%.+]] = [[CST_0_]]) -> (index, index)  : i32 {
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addi [[VAR_arg5_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.addi [[VAR_arg6_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]]:2 = scf.for [[VAR_arg7_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_1_]] iter_args([[VAR_arg8_:%.+]] = [[VAR_8_]], [[VAR_arg9_:%.+]] = [[VAR_9_]]) -> (index, index)  : i32 {
// CHECK-DAG:           [[VAR_15_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, offsets: {{.}}[[VAR_arg9_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:           [[VAR_16_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_]], [[VAR_1_]]{{.}}, offsets: {{.}}[[VAR_arg8_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:               [[VAR_17_:%.+]] = "tts.load"([[VAR_16_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:               "tts.store"([[VAR_15_]], [[VAR_17_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK-DAG:           [[VAR_18_:%.+]] = arith.addi [[VAR_arg8_]], [[CST_3_]] : index
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.addi [[VAR_arg9_]], [[CST_3_]] : index
// CHECK:               scf.yield [[VAR_18_]], [[VAR_19_]] : index, index
// CHECK:             }
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[VAR_arg5_]], [[VAR_6_]] : index
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.addi [[VAR_11_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addi [[VAR_arg6_]], [[VAR_5_]] : index
// CHECK:             [[VAR_14_:%.+]] = arith.addi [[VAR_13_]], [[CST_1_]] : index
// CHECK:             scf.yield [[VAR_12_]], [[VAR_14_]] : index, index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
//
// CHECK:         tt.func public @nested2_use_loop_results([[arg0_:.+]]: !tt.ptr<f32>, [[arg1_:.+]]: !tt.ptr<f32>, [[arg2_:.+]]: i32, [[arg3_:.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[VAR_0_1_:%.+]] = arith.index_cast [[arg2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_1_:%.+]] = arith.index_cast [[arg3_]] : i32 to index
// CHECK-DAG:       [[VAR_2_1_:%.+]] = arith.index_cast [[arg2_]] : i32 to index
// CHECK-DAG:       [[VAR_3_1_:%.+]] = arith.index_cast [[arg3_]] : i32 to index
// CHECK:           [[VAR_4_1_:%.+]] = arith.muli [[arg3_]], [[CST_4_]] : i32
// CHECK-DAG:       [[VAR_5_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : i32 to index
// CHECK-DAG:       [[VAR_6_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : i32 to index
// CHECK-DAG:       [[VAR_7_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : i32 to index
// CHECK-DAG:       [[VAR_8_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : i32 to index
// CHECK-DAG:       [[VAR_9_1_:%.+]]:2 = scf.for [[VAR_arg4_1_:%.+]] = [[CST_0_3_]] to [[CST_2_1_]] step [[CST_1_2_]] iter_args([[VAR_arg5_1_:%.+]] = [[CST_0_2_]], [[VAR_arg6_1_:%.+]] = [[CST_0_2_]]) -> (index, index)  : i32 {
// CHECK-DAG:         [[VAR_10_1_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_1_]], [[VAR_3_1_]]{{.}}, offsets: {{.}}[[VAR_arg6_1_]], [[CST_0_2_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_11_1_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_1_]], [[VAR_1_1_]]{{.}}, offsets: {{.}}[[VAR_arg5_1_]], [[CST_0_2_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:             [[VAR_12_1_:%.+]] = "tts.load"([[VAR_11_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:             "tts.store"([[VAR_10_1_]], [[VAR_12_1_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK-DAG:         [[VAR_13_1_:%.+]] = arith.addi [[VAR_arg5_1_]], [[VAR_8_1_]] : index
// CHECK-DAG:         [[VAR_14_1_:%.+]] = arith.addi [[VAR_arg6_1_]], [[VAR_7_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_15_1_:%.+]]:2 = scf.for [[VAR_arg7_1_:%.+]] = [[CST_0_3_]] to [[CST_2_1_]] step [[CST_1_2_]] iter_args([[VAR_arg8_1_:%.+]] = [[VAR_13_1_]], [[VAR_arg9_1_:%.+]] = [[VAR_14_1_]]) -> (index, index)  : i32 {
// CHECK-DAG:           [[VAR_16_1_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_1_]], [[VAR_3_1_]]{{.}}, offsets: {{.}}[[VAR_arg9_1_]], [[CST_0_2_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:           [[VAR_17_1_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_1_]], [[VAR_1_1_]]{{.}}, offsets: {{.}}[[VAR_arg8_1_]], [[CST_0_2_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:               [[VAR_18_1_:%.+]] = "tts.load"([[VAR_17_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:               "tts.store"([[VAR_16_1_]], [[VAR_18_1_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK-DAG:           [[VAR_19_1_:%.+]] = arith.addi [[VAR_arg8_1_]], [[VAR_6_1_]] : index
// CHECK-DAG:           [[VAR_20_:%.+]] = arith.addi [[VAR_arg9_1_]], [[VAR_5_1_]] : index
// CHECK:               scf.yield [[VAR_19_1_]], [[VAR_20_]] : index, index
// CHECK:             }
// CHECK:             scf.yield [[VAR_15_1_]]#0, [[VAR_15_1_]]#1 : index, index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
//
// CHECK:         tt.func public @nested3([[arg0_:.+]]: !tt.ptr<f32>, [[arg1_:.+]]: !tt.ptr<f32>, [[arg2_:.+]]: i32, [[arg3_:.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_3_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_5_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[VAR_0_2_:%.+]] = arith.index_cast [[arg2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_2_:%.+]] = arith.index_cast [[arg3_]] : i32 to index
// CHECK-DAG:       [[VAR_2_2_:%.+]] = arith.index_cast [[arg2_]] : i32 to index
// CHECK-DAG:       [[VAR_3_2_:%.+]] = arith.index_cast [[arg3_]] : i32 to index
// CHECK:           [[VAR_4_2_:%.+]] = arith.muli [[arg3_]], [[CST_2_2_]] : i32
// CHECK-DAG:       [[VAR_5_2_:%.+]] = arith.index_cast [[VAR_4_2_]] : i32 to index
// CHECK-DAG:       [[VAR_6_2_:%.+]] = arith.index_cast [[VAR_4_2_]] : i32 to index
// CHECK-DAG:       [[VAR_7_2_:%.+]] = arith.index_cast [[VAR_4_2_]] : i32 to index
// CHECK-DAG:       [[VAR_8_2_:%.+]] = arith.index_cast [[VAR_4_2_]] : i32 to index
// CHECK-DAG:       [[VAR_9_2_:%.+]] = arith.index_cast [[VAR_4_2_]] : i32 to index
// CHECK-DAG:       [[VAR_10_2_:%.+]] = arith.muli [[arg3_]], [[CST_2_2_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_2_:%.+]] = arith.index_cast [[VAR_10_2_]] : i32 to index
// CHECK-DAG:       [[VAR_12_2_:%.+]]:2 = scf.for [[VAR_arg4_2_:%.+]] = [[CST_0_5_]] to [[CST_2_2_]] step [[CST_1_3_]] iter_args([[VAR_arg5_2_:%.+]] = [[CST_0_4_]], [[VAR_arg6_2_:%.+]] = [[CST_0_4_]]) -> (index, index)  : i32 {
// CHECK-DAG:         [[VAR_13_2_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_2_]], [[VAR_1_2_]]{{.}}, offsets: {{.}}[[VAR_arg5_2_]], [[CST_0_4_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_2_:%.+]] = "tts.load"([[VAR_13_2_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:         [[VAR_15_2_:%.+]]:2 = scf.for [[VAR_arg7_2_:%.+]] = [[CST_0_5_]] to [[CST_2_2_]] step [[CST_1_3_]] iter_args([[VAR_arg8_2_:%.+]] = [[VAR_arg5_2_]], [[VAR_arg9_2_:%.+]] = [[VAR_arg6_2_]]) -> (index, index)  : i32 {
// CHECK-DAG:           [[VAR_17_2_:%.+]] = arith.addi [[VAR_arg8_2_]], [[VAR_9_2_]] : index
// CHECK:               [[VAR_18_2_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_2_]], [[VAR_1_2_]]{{.}}, offsets: {{.}}[[VAR_17_2_]], [[CST_0_4_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:           [[VAR_19_2_:%.+]] = "tts.load"([[VAR_18_2_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:           [[VAR_20_1_:%.+]]:2 = scf.for [[VAR_arg10_:%.+]] = [[CST_0_5_]] to [[CST_2_2_]] step [[CST_1_3_]] iter_args([[VAR_arg11_:%.+]] = [[VAR_17_2_]], [[VAR_arg12_:%.+]] = [[VAR_arg9_2_]]) -> (index, index)  : i32 {
// CHECK-DAG:             [[VAR_21_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_2_]], [[VAR_3_2_]]{{.}}, offsets: {{.}}[[VAR_arg12_]], [[CST_0_4_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:             [[VAR_22_:%.+]] = arith.addi [[VAR_arg11_]], [[VAR_8_2_]] : index
// CHECK:                 [[VAR_23_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_2_]], [[VAR_1_2_]]{{.}}, offsets: {{.}}[[VAR_22_]], [[CST_0_4_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                 [[VAR_24_:%.+]] = "tts.load"([[VAR_23_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:                 "tts.store"([[VAR_21_]], [[VAR_14_2_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                 [[VAR_25_:%.+]] = arith.addi [[VAR_arg12_]], [[VAR_7_2_]] : index
// CHECK:                 [[VAR_26_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_2_]], [[VAR_3_2_]]{{.}}, offsets: {{.}}[[VAR_25_]], [[CST_0_4_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                 "tts.store"([[VAR_26_]], [[VAR_19_2_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                 [[VAR_27_:%.+]] = arith.addi [[VAR_25_]], [[VAR_6_2_]] : index
// CHECK:                 [[VAR_28_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_2_]], [[VAR_3_2_]]{{.}}, offsets: {{.}}[[VAR_27_]], [[CST_0_4_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:                 "tts.store"([[VAR_28_]], [[VAR_24_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:                 [[VAR_29_:%.+]] = arith.addi [[VAR_27_]], [[VAR_5_2_]] : index
// CHECK:                 scf.yield [[VAR_22_]], [[VAR_29_]] : index, index
// CHECK:               }
// CHECK:               scf.yield [[VAR_20_1_]]#0, [[VAR_20_1_]]#1 : index, index
// CHECK:             }
// CHECK:             [[VAR_16_2_:%.+]] = arith.addi [[VAR_15_2_]]#0, [[VAR_11_2_]] : index
// CHECK:             scf.yield [[VAR_16_2_]], [[VAR_15_2_]]#1 : index, index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
//
// CHECK:         tt.func public @nested_use_same_level_loop_result([[arg0_:.+]]: !tt.ptr<f32>, [[arg1_:.+]]: !tt.ptr<f32>, [[arg2_:.+]]: i32, [[arg3_:.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_4_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_7_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2_3_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[VAR_0_3_:%.+]] = arith.index_cast [[arg2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_3_:%.+]] = arith.index_cast [[arg3_]] : i32 to index
// CHECK-DAG:       [[VAR_2_3_:%.+]] = arith.index_cast [[arg2_]] : i32 to index
// CHECK-DAG:       [[VAR_3_3_:%.+]] = arith.index_cast [[arg3_]] : i32 to index
// CHECK:           [[VAR_4_3_:%.+]] = arith.muli [[arg3_]], [[CST_2_3_]] : i32
// CHECK-DAG:       [[VAR_5_3_:%.+]] = arith.index_cast [[VAR_4_3_]] : i32 to index
// CHECK-DAG:       [[VAR_6_3_:%.+]] = arith.muli [[arg3_]], [[CST_2_3_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_3_:%.+]] = arith.index_cast [[VAR_6_3_]] : i32 to index
// CHECK-DAG:       [[VAR_8_3_:%.+]] = arith.index_cast [[VAR_6_3_]] : i32 to index
// CHECK-DAG:       [[VAR_9_3_:%.+]] = arith.index_cast [[VAR_6_3_]] : i32 to index
// CHECK-DAG:       [[VAR_10_3_:%.+]] = arith.index_cast [[VAR_6_3_]] : i32 to index
// CHECK-DAG:       [[VAR_11_3_:%.+]] = arith.index_cast [[VAR_6_3_]] : i32 to index
// CHECK-DAG:       [[VAR_12_3_:%.+]] = arith.muli [[arg3_]], [[CST_2_3_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_3_:%.+]] = arith.index_cast [[VAR_12_3_]] : i32 to index
// CHECK-DAG:       [[VAR_14_3_:%.+]]:2 = scf.for [[VAR_arg4_3_:%.+]] = [[CST_0_7_]] to [[CST_2_3_]] step [[CST_1_4_]] iter_args([[VAR_arg5_3_:%.+]] = [[CST_0_6_]], [[VAR_arg6_3_:%.+]] = [[CST_0_6_]]) -> (index, index)  : i32 {
// CHECK-DAG:         [[VAR_15_3_:%.+]] = scf.for [[VAR_arg7_3_:%.+]] = [[CST_0_7_]] to [[CST_2_3_]] step [[CST_1_4_]] iter_args([[VAR_arg8_3_:%.+]] = [[VAR_arg5_3_]]) -> (index)  : i32 {
// CHECK-DAG:           [[VAR_18_3_:%.+]] = arith.addi [[VAR_arg8_3_]], [[VAR_5_3_]] : index
// CHECK:               scf.yield [[VAR_18_3_]] : index
// CHECK:             }
// CHECK-DAG:         [[VAR_16_3_:%.+]]:2 = scf.for [[VAR_arg7_4_:%.+]] = [[CST_0_7_]] to [[CST_2_3_]] step [[CST_1_4_]] iter_args([[VAR_arg8_4_:%.+]] = [[VAR_15_3_]], [[VAR_arg9_3_:%.+]] = [[VAR_arg6_3_]]) -> (index, index)  : i32 {
// CHECK-DAG:           [[VAR_18_4_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_3_]], [[VAR_3_3_]]{{.}}, offsets: {{.}}[[VAR_arg9_3_]], [[CST_0_6_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-DAG:           [[VAR_19_3_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_3_]], [[VAR_1_3_]]{{.}}, offsets: {{.}}[[VAR_arg8_4_]], [[CST_0_6_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_20_2_:%.+]] = "tts.load"([[VAR_19_3_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK-DAG:           [[VAR_21_1_:%.+]] = arith.addi [[VAR_arg8_4_]], [[VAR_11_3_]] : index
// CHECK:               [[VAR_22_1_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [2, 2], strides: {{.}}[[VAR_0_3_]], [[VAR_1_3_]]{{.}}, offsets: {{.}}[[VAR_21_1_]], [[CST_0_6_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:               [[VAR_23_1_:%.+]] = "tts.load"([[VAR_22_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>) -> tensor<2x2xf32>
// CHECK:               "tts.store"([[VAR_18_4_]], [[VAR_20_2_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK:               [[VAR_24_1_:%.+]] = arith.addi [[VAR_arg9_3_]], [[VAR_10_3_]] : index
// CHECK:               [[VAR_25_1_:%.+]] = arith.addi [[VAR_24_1_]], [[VAR_9_3_]] : index
// CHECK:               [[VAR_26_1_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [2, 2], strides: {{.}}[[VAR_2_3_]], [[VAR_3_3_]]{{.}}, offsets: {{.}}[[VAR_25_1_]], [[CST_0_6_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<2x2x!tt.ptr<f32>>
// CHECK:               "tts.store"([[VAR_26_1_]], [[VAR_23_1_]]) <{static_mask_dims = array<i64>}> : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>) -> ()
// CHECK-DAG:           [[VAR_27_1_:%.+]] = arith.addi [[VAR_25_1_]], [[VAR_8_3_]] : index
// CHECK-DAG:           [[VAR_28_1_:%.+]] = arith.addi [[VAR_21_1_]], [[VAR_7_3_]] : index
// CHECK:               scf.yield [[VAR_28_1_]], [[VAR_27_1_]] : index, index
// CHECK:             }
// CHECK:             [[VAR_17_3_:%.+]] = arith.addi [[VAR_16_3_]]#0, [[VAR_13_3_]] : index
// CHECK:             scf.yield [[VAR_17_3_]], [[VAR_16_3_]]#1 : index, index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
