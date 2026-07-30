// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

module {
  tt.func public @_layer_norm_fwd_fused_0123456789(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>, %arg6: i32, %arg7: i32, %arg8: f32) {
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg6 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %cst_0 : f32 -> tensor<256xf32>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %6 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %7 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg10 = %4) -> (tensor<256xf32>)  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %5 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %6 : tensor<256xi32>
      %35 = tt.addptr %7, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34, %4 : tensor<256x!tt.ptr<f32>>
      %37 = arith.addf %arg10, %36 : tensor<256xf32>
      scf.yield %37 : tensor<256xf32>
    }
    %9 = "tt.reduce"(%8) ({
    ^bb0(%arg9: f32, %arg10: f32):
      %32 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %32 : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %10 = arith.sitofp %arg7 : i32 to f32
    %11 = arith.divf %9, %10 : f32
    %12 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %13 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %14 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %15 = tt.splat %11 : f32 -> tensor<256xf32>
    %16 = scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg10 = %4) -> (tensor<256xf32>)  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %12 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %13 : tensor<256xi32>
      %35 = tt.addptr %14, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34, %4 : tensor<256x!tt.ptr<f32>>
      %37 = arith.subf %36, %15 : tensor<256xf32>
      %38 = arith.select %34, %37, %4 : tensor<256xi1>, tensor<256xf32>
      %39 = arith.mulf %38, %38 : tensor<256xf32>
      %40 = arith.addf %arg10, %39 : tensor<256xf32>
      scf.yield %40 : tensor<256xf32>
    }
    %17 = "tt.reduce"(%16) ({
    ^bb0(%arg9: f32, %arg10: f32):
      %32 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %32 : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %18 = arith.divf %17, %10 : f32
    %19 = arith.addf %18, %arg8 : f32
    %20 = math.sqrt %19 : f32
    %21 = arith.divf %cst, %20 : f32
    %22 = tt.addptr %arg4, %0 : !tt.ptr<f32>, i32
    tt.store %22, %11 : !tt.ptr<f32>
    %23 = tt.addptr %arg5, %0 : !tt.ptr<f32>, i32
    tt.store %23, %21 : !tt.ptr<f32>
    %24 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %25 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %26 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %27 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %28 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %29 = tt.splat %11 : f32 -> tensor<256xf32>
    %30 = tt.splat %21 : f32 -> tensor<256xf32>
    %31 = tt.splat %2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %24 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %25 : tensor<256xi32>
      %35 = tt.addptr %26, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34 : tensor<256x!tt.ptr<f32>>
      %37 = tt.addptr %27, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %38 = tt.load %37, %34 : tensor<256x!tt.ptr<f32>>
      %39 = tt.addptr %28, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %40 = tt.load %39, %34, %4 : tensor<256x!tt.ptr<f32>>
      %41 = arith.subf %40, %29 : tensor<256xf32>
      %42 = arith.mulf %41, %30 : tensor<256xf32>
      %43 = arith.mulf %42, %36 : tensor<256xf32>
      %44 = arith.addf %43, %38 : tensor<256xf32>
      %45 = tt.addptr %31, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      tt.store %45, %44, %34 : tensor<256x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK:         tt.func public @_layer_norm_fwd_fused_0123456789([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: !tt.ptr<f32>, [[PARAM_3_:%.+]]: !tt.ptr<f32>, [[PARAM_4_:%.+]]: !tt.ptr<f32>, [[PARAM_5_:%.+]]: !tt.ptr<f32>, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: f32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : tensor<256xf32>
// CHECK-DAG:       [[CST_256_1_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_6_]] : i32
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = scf.for [[VAR_arg9_:%.+]] = [[CST_0_]] to [[PARAM_7_]] step [[CST_256_1_]] iter_args([[VAR_arg10_:%.+]] = [[VAR_cst_0_]]) -> (tensor<256xf32>)  : i32 {
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.index_cast [[VAR_arg9_]] : i32 to index
// CHECK:             [[VAR_22_:%.+]] = arith.addi [[VAR_2_]], [[VAR_21_]] : index
// CHECK-DAG:         [[VAR_23_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [256], strides: [1], offsets: {{.}}[[VAR_22_]]{{.}}, shape: [0], order: [] : <f32> to tensor<256x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.index_cast [[VAR_arg9_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.addi [[VAR_24_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_27_:%.+]] = arith.minsi [[VAR_25_]], [[VAR_26_]] : index
// CHECK:             [[VAR_28_:%.+]] = arith.maxsi [[VAR_27_]], [[VAR_24_]] : index
// CHECK:             [[VAR_29_:%.+]] = arith.subi [[VAR_28_]], [[VAR_24_]] : index
// CHECK:             [[VAR_30_:%.+]] = "tts.load"([[VAR_23_]], [[VAR_29_]], [[CST_0_dot_000000_]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<256x!tt.ptr<f32>>, index, f32) -> tensor<256xf32>
// CHECK:             [[VAR_31_:%.+]] = arith.addf [[VAR_arg10_]], [[VAR_30_]] : tensor<256xf32>
// CHECK:             scf.yield [[VAR_31_]] : tensor<256xf32>
// CHECK:           }
// CHECK:           [[VAR_5_:%.+]] = "tt.reduce"([[VAR_4_]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0([[VAR_arg9_1_:%.+]]: f32, [[VAR_arg10_1_:%.+]]: f32):
// CHECK:             [[VAR_21_1_:%.+]] = arith.addf [[VAR_arg9_1_]], [[VAR_arg10_1_]] : f32
// CHECK:             tt.reduce.return [[VAR_21_1_]] : f32
// CHECK:           }) : (tensor<256xf32>) -> f32
// CHECK:           [[VAR_6_:%.+]] = arith.sitofp [[PARAM_7_]] : i32 to f32
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.divf [[VAR_5_]], [[VAR_6_]] : f32
// CHECK-DAG:       [[VAR_8_:%.+]] = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tt.splat [[PARAM_7_]] : i32 -> tensor<256xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = tt.splat [[VAR_7_]] : f32 -> tensor<256xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = scf.for [[VAR_arg9_2_:%.+]] = [[CST_0_]] to [[PARAM_7_]] step [[CST_256_1_]] iter_args([[VAR_arg10_2_:%.+]] = [[VAR_cst_0_]]) -> (tensor<256xf32>)  : i32 {
// CHECK-DAG:         [[VAR_21_2_:%.+]] = tt.splat [[VAR_arg9_2_]] : i32 -> tensor<256xi32>
// CHECK:             [[VAR_22_1_:%.+]] = arith.addi [[VAR_21_2_]], [[VAR_8_]] : tensor<256xi32>
// CHECK-DAG:         [[VAR_23_1_:%.+]] = arith.cmpi slt, [[VAR_22_1_]], [[VAR_9_]] : tensor<256xi32>
// CHECK-DAG:         [[VAR_24_1_:%.+]] = arith.index_cast [[VAR_arg9_2_]] : i32 to index
// CHECK:             [[VAR_25_1_:%.+]] = arith.addi [[VAR_2_]], [[VAR_24_1_]] : index
// CHECK-DAG:         [[VAR_26_1_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [256], strides: [1], offsets: {{.}}[[VAR_25_1_]]{{.}}, shape: [0], order: [] : <f32> to tensor<256x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_27_1_:%.+]] = arith.index_cast [[VAR_arg9_2_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_28_1_:%.+]] = arith.addi [[VAR_27_1_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_29_1_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_30_1_:%.+]] = arith.minsi [[VAR_28_1_]], [[VAR_29_1_]] : index
// CHECK:             [[VAR_31_1_:%.+]] = arith.maxsi [[VAR_30_1_]], [[VAR_27_1_]] : index
// CHECK:             [[VAR_32_:%.+]] = arith.subi [[VAR_31_1_]], [[VAR_27_1_]] : index
// CHECK:             [[VAR_33_:%.+]] = "tts.load"([[VAR_26_1_]], [[VAR_32_]], [[CST_0_dot_000000_]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<256x!tt.ptr<f32>>, index, f32) -> tensor<256xf32>
// CHECK:             [[VAR_34_:%.+]] = arith.subf [[VAR_33_]], [[VAR_10_]] : tensor<256xf32>
// CHECK:             [[VAR_35_:%.+]] = arith.select [[VAR_23_1_]], [[VAR_34_]], [[VAR_cst_0_]] : tensor<256xi1>, tensor<256xf32>
// CHECK:             [[VAR_36_:%.+]] = arith.mulf [[VAR_35_]], [[VAR_35_]] : tensor<256xf32>
// CHECK:             [[VAR_37_:%.+]] = arith.addf [[VAR_arg10_2_]], [[VAR_36_]] : tensor<256xf32>
// CHECK:             scf.yield [[VAR_37_]] : tensor<256xf32>
// CHECK:           }
// CHECK:           [[VAR_12_:%.+]] = "tt.reduce"([[VAR_11_]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0([[VAR_arg9_2_:%.+]]: f32, [[VAR_arg10_2_:%.+]]: f32):
// CHECK:             [[VAR_21_3_:%.+]] = arith.addf [[VAR_arg9_2_]], [[VAR_arg10_2_]] : f32
// CHECK:             tt.reduce.return [[VAR_21_3_]] : f32
// CHECK:           }) : (tensor<256xf32>) -> f32
// CHECK:           [[VAR_13_:%.+]] = arith.divf [[VAR_12_]], [[VAR_6_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.addf [[VAR_13_]], [[PARAM_8_]] : f32
// CHECK:           [[VAR_15_:%.+]] = math.sqrt [[VAR_14_]] : f32
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_15_]] : f32
// CHECK-DAG:       [[VAR_17_:%.+]] = tt.addptr [[PARAM_4_]], [[VAR_0_]] : !tt.ptr<f32>, i32
// CHECK:           tt.store [[VAR_17_]], [[VAR_7_]] : !tt.ptr<f32>
// CHECK:           [[VAR_18_:%.+]] = tt.addptr [[PARAM_5_]], [[VAR_0_]] : !tt.ptr<f32>, i32
// CHECK:           tt.store [[VAR_18_]], [[VAR_16_]] : !tt.ptr<f32>
// CHECK-DAG:       [[VAR_19_:%.+]] = tt.splat [[VAR_7_]] : f32 -> tensor<256xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = tt.splat [[VAR_16_]] : f32 -> tensor<256xf32>
// CHECK:           scf.for [[VAR_arg9_2_1_:%.+]] = [[CST_0_]] to [[PARAM_7_]] step [[CST_256_1_]]  : i32 {
// CHECK:             [[VAR_21_4_:%.+]] = arith.index_cast [[VAR_arg9_2_1_]] : i32 to index
// CHECK-DAG:         [[VAR_22_2_:%.+]] = tts.make_tptr [[PARAM_2_]] to sizes: [256], strides: [1], offsets: {{.}}[[VAR_21_4_]]{{.}}, shape: [0], order: [] : <f32> to tensor<256x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_23_2_:%.+]] = arith.index_cast [[VAR_arg9_2_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_24_2_:%.+]] = arith.addi [[VAR_23_2_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_25_2_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_26_2_:%.+]] = arith.minsi [[VAR_24_2_]], [[VAR_25_2_]] : index
// CHECK:             [[VAR_27_2_:%.+]] = arith.maxsi [[VAR_26_2_]], [[VAR_23_2_]] : index
// CHECK:             [[VAR_28_2_:%.+]] = arith.subi [[VAR_27_2_]], [[VAR_23_2_]] : index
// CHECK-DAG:         [[VAR_29_2_:%.+]] = "tts.load"([[VAR_22_2_]], [[VAR_28_2_]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<256x!tt.ptr<f32>>, index) -> tensor<256xf32>
// CHECK-DAG:         [[VAR_30_2_:%.+]] = arith.index_cast [[VAR_arg9_2_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_31_2_:%.+]] = tts.make_tptr [[PARAM_3_]] to sizes: [256], strides: [1], offsets: {{.}}[[VAR_30_2_]]{{.}}, shape: [0], order: [] : <f32> to tensor<256x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_32_1_:%.+]] = arith.index_cast [[VAR_arg9_2_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_33_1_:%.+]] = arith.addi [[VAR_32_1_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_34_1_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_35_1_:%.+]] = arith.minsi [[VAR_33_1_]], [[VAR_34_1_]] : index
// CHECK:             [[VAR_36_1_:%.+]] = arith.maxsi [[VAR_35_1_]], [[VAR_32_1_]] : index
// CHECK:             [[VAR_37_1_:%.+]] = arith.subi [[VAR_36_1_]], [[VAR_32_1_]] : index
// CHECK-DAG:         [[VAR_38_:%.+]] = "tts.load"([[VAR_31_2_]], [[VAR_37_1_]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<256x!tt.ptr<f32>>, index) -> tensor<256xf32>
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.index_cast [[VAR_arg9_2_1_]] : i32 to index
// CHECK:             [[VAR_40_:%.+]] = arith.addi [[VAR_2_]], [[VAR_39_]] : index
// CHECK-DAG:         [[VAR_41_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [256], strides: [1], offsets: {{.}}[[VAR_40_]]{{.}}, shape: [0], order: [] : <f32> to tensor<256x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_42_:%.+]] = arith.index_cast [[VAR_arg9_2_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_43_:%.+]] = arith.addi [[VAR_42_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_45_:%.+]] = arith.minsi [[VAR_43_]], [[VAR_44_]] : index
// CHECK:             [[VAR_46_:%.+]] = arith.maxsi [[VAR_45_]], [[VAR_42_]] : index
// CHECK:             [[VAR_47_:%.+]] = arith.subi [[VAR_46_]], [[VAR_42_]] : index
// CHECK:             [[VAR_48_:%.+]] = "tts.load"([[VAR_41_]], [[VAR_47_]], [[CST_0_dot_000000_]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<256x!tt.ptr<f32>>, index, f32) -> tensor<256xf32>
// CHECK:             [[VAR_49_:%.+]] = arith.subf [[VAR_48_]], [[VAR_19_]] : tensor<256xf32>
// CHECK:             [[VAR_50_:%.+]] = arith.mulf [[VAR_49_]], [[VAR_20_]] : tensor<256xf32>
// CHECK:             [[VAR_51_:%.+]] = arith.mulf [[VAR_50_]], [[VAR_29_2_]] : tensor<256xf32>
// CHECK-DAG:         [[VAR_52_:%.+]] = arith.addf [[VAR_51_]], [[VAR_38_]] : tensor<256xf32>
// CHECK-DAG:         [[VAR_53_:%.+]] = arith.index_cast [[VAR_arg9_2_1_]] : i32 to index
// CHECK:             [[VAR_54_:%.+]] = arith.addi [[VAR_3_]], [[VAR_53_]] : index
// CHECK-DAG:         [[VAR_55_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [256], strides: [1], offsets: {{.}}[[VAR_54_]]{{.}}, shape: [0], order: [] : <f32> to tensor<256x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_56_:%.+]] = arith.index_cast [[VAR_arg9_2_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_57_:%.+]] = arith.addi [[VAR_56_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_58_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:             [[VAR_59_:%.+]] = arith.minsi [[VAR_57_]], [[VAR_58_]] : index
// CHECK:             [[VAR_60_:%.+]] = arith.maxsi [[VAR_59_]], [[VAR_56_]] : index
// CHECK:             [[VAR_61_:%.+]] = arith.subi [[VAR_60_]], [[VAR_56_]] : index
// CHECK:             "tts.store"([[VAR_55_]], [[VAR_52_]], [[VAR_61_]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<256x!tt.ptr<f32>>, tensor<256xf32>, index) -> ()
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
