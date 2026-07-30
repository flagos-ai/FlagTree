// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

module {
  tt.func public @_layer_norm_bwd_dwdb_0123456(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: i32, %arg5: i32) {
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c256_i32 : i32
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : i32 -> tensor<256xi32>
    %4 = arith.addi %3, %2 : tensor<256xi32>
    %5 = tt.splat %cst : f32 -> tensor<256x256xf32>
    %6 = tt.splat %arg4 : i32 -> tensor<256x1xi32>
    %7 = tt.expand_dims %4 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %8 = tt.splat %arg5 : i32 -> tensor<1x256xi32>
    %9 = arith.cmpi slt, %7, %8 : tensor<1x256xi32>
    %10 = tt.broadcast %9 : tensor<1x256xi1> -> tensor<256x256xi1>
    %11 = tt.splat %arg5 : i32 -> tensor<256x1xi32>
    %12 = tt.broadcast %7 : tensor<1x256xi32> -> tensor<256x256xi32>
    %13 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x256x!tt.ptr<f32>>
    %14 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x256x!tt.ptr<f32>>
    %15:2 = scf.for %arg6 = %c0_i32 to %arg4 step %c256_i32 iter_args(%arg7 = %5, %arg8 = %5) -> (tensor<256x256xf32>, tensor<256x256xf32>)  : i32 {
      %24 = tt.splat %arg6 : i32 -> tensor<256xi32>
      %25 = arith.addi %24, %2 : tensor<256xi32>
      %26 = tt.expand_dims %25 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
      %27 = arith.cmpi slt, %26, %6 : tensor<256x1xi32>
      %28 = tt.broadcast %27 : tensor<256x1xi1> -> tensor<256x256xi1>
      %29 = arith.andi %28, %10 : tensor<256x256xi1>
      %30 = arith.muli %26, %11 : tensor<256x1xi32>
      %31 = tt.broadcast %30 : tensor<256x1xi32> -> tensor<256x256xi32>
      %32 = arith.addi %31, %12 : tensor<256x256xi32>
      %33 = tt.addptr %13, %32 : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
      %34 = tt.load %33, %29, %5 : tensor<256x256x!tt.ptr<f32>>
      %35 = arith.addf %arg7, %34 : tensor<256x256xf32>
      %36 = tt.addptr %14, %32 : tensor<256x256x!tt.ptr<f32>>, tensor<256x256xi32>
      %37 = tt.load %36, %29, %5 : tensor<256x256x!tt.ptr<f32>>
      %38 = arith.addf %arg8, %37 : tensor<256x256xf32>
      scf.yield %35, %38 : tensor<256x256xf32>, tensor<256x256xf32>
    }
    %16 = "tt.reduce"(%15#0) ({
    ^bb0(%arg6: f32, %arg7: f32):
      %24 = arith.addf %arg6, %arg7 : f32
      tt.reduce.return %24 : f32
    }) {axis = 0 : i32} : (tensor<256x256xf32>) -> tensor<256xf32>
    %17 = "tt.reduce"(%15#1) ({
    ^bb0(%arg6: f32, %arg7: f32):
      %24 = arith.addf %arg6, %arg7 : f32
      tt.reduce.return %24 : f32
    }) {axis = 0 : i32} : (tensor<256x256xf32>) -> tensor<256xf32>
    %18 = tt.splat %arg5 : i32 -> tensor<256xi32>
    %19 = arith.cmpi slt, %4, %18 : tensor<256xi32>
    %20 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %21 = tt.addptr %20, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %21, %16, %19 : tensor<256x!tt.ptr<f32>>
    %22 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %23 = tt.addptr %22, %4 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %23, %17, %19 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func public @_layer_norm_bwd_dwdb_0123456([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: !tt.ptr<f32>, [[PARAM_3_:%.+]]: !tt.ptr<f32>, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : tensor<256x256xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_256_1_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_256_1_]] : i32
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_6_:%.+]]:2 = scf.for [[VAR_arg6_:%.+]] = [[CST_0_]] to [[PARAM_4_]] step [[CST_256_1_]] iter_args([[VAR_arg7_:%.+]] = [[VAR_cst_0_]], [[VAR_arg8_:%.+]] = [[VAR_cst_0_]]) -> (tensor<256x256xf32>, tensor<256x256xf32>)  : i32 {
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.index_cast [[VAR_arg6_]] : i32 to index
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:             [[VAR_25_:%.+]] = arith.muli [[VAR_23_]], [[VAR_24_]] : index
// CHECK-DAG:         [[VAR_26_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [256, 256], strides: {{.}}[[VAR_24_]], 1], offsets: {{.}}[[VAR_25_]], [[VAR_5_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<256x256x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.index_cast [[VAR_arg6_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.addi [[VAR_27_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:             [[VAR_30_:%.+]] = arith.minsi [[VAR_28_]], [[VAR_29_]] : index
// CHECK:             [[VAR_31_:%.+]] = arith.maxsi [[VAR_30_]], [[VAR_27_]] : index
// CHECK-DAG:         [[VAR_32_:%.+]] = arith.subi [[VAR_31_]], [[VAR_27_]] : index
// CHECK-DAG:         [[VAR_33_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.addi [[VAR_33_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:             [[VAR_36_:%.+]] = arith.minsi [[VAR_34_]], [[VAR_35_]] : index
// CHECK:             [[VAR_37_:%.+]] = arith.maxsi [[VAR_36_]], [[VAR_33_]] : index
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.subi [[VAR_37_]], [[VAR_33_]] : index
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.minsi [[VAR_32_]], [[CST_256_]] : index
// CHECK:             [[VAR_40_:%.+]] = arith.minsi [[VAR_38_]], [[CST_256_]] : index
// CHECK:             [[VAR_41_:%.+]] = "tts.load"([[VAR_26_]], [[VAR_39_]], [[VAR_40_]], [[CST_0_dot_000000_]]) <{operandSegmentSizes = array<i32: 1, 2, 1>, static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808>}> : (tensor<256x256x!tt.ptr<f32>>, index, index, f32) -> tensor<256x256xf32>
// CHECK-DAG:         [[VAR_42_:%.+]] = arith.addf [[VAR_arg7_]], [[VAR_41_]] : tensor<256x256xf32>
// CHECK-DAG:         [[VAR_43_:%.+]] = arith.index_cast [[VAR_arg6_]] : i32 to index
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:             [[VAR_45_:%.+]] = arith.muli [[VAR_43_]], [[VAR_44_]] : index
// CHECK-DAG:         [[VAR_46_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [256, 256], strides: {{.}}[[VAR_44_]], 1], offsets: {{.}}[[VAR_45_]], [[VAR_4_]]{{.}}, shape: [0, 0], order: [] : <f32> to tensor<256x256x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.index_cast [[VAR_arg6_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.addi [[VAR_47_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_49_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:             [[VAR_50_:%.+]] = arith.minsi [[VAR_48_]], [[VAR_49_]] : index
// CHECK:             [[VAR_51_:%.+]] = arith.maxsi [[VAR_50_]], [[VAR_47_]] : index
// CHECK-DAG:         [[VAR_52_:%.+]] = arith.subi [[VAR_51_]], [[VAR_47_]] : index
// CHECK-DAG:         [[VAR_53_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_54_:%.+]] = arith.addi [[VAR_53_]], [[CST_256_]] : index
// CHECK-DAG:         [[VAR_55_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:             [[VAR_56_:%.+]] = arith.minsi [[VAR_54_]], [[VAR_55_]] : index
// CHECK:             [[VAR_57_:%.+]] = arith.maxsi [[VAR_56_]], [[VAR_53_]] : index
// CHECK-DAG:         [[VAR_58_:%.+]] = arith.subi [[VAR_57_]], [[VAR_53_]] : index
// CHECK-DAG:         [[VAR_59_:%.+]] = arith.minsi [[VAR_52_]], [[CST_256_]] : index
// CHECK:             [[VAR_60_:%.+]] = arith.minsi [[VAR_58_]], [[CST_256_]] : index
// CHECK:             [[VAR_61_:%.+]] = "tts.load"([[VAR_46_]], [[VAR_59_]], [[VAR_60_]], [[CST_0_dot_000000_]]) <{operandSegmentSizes = array<i32: 1, 2, 1>, static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808>}> : (tensor<256x256x!tt.ptr<f32>>, index, index, f32) -> tensor<256x256xf32>
// CHECK:             [[VAR_62_:%.+]] = arith.addf [[VAR_arg8_]], [[VAR_61_]] : tensor<256x256xf32>
// CHECK:             scf.yield [[VAR_42_]], [[VAR_62_]] : tensor<256x256xf32>, tensor<256x256xf32>
// CHECK:           }
// CHECK:           [[VAR_7_:%.+]] = "tt.reduce"([[VAR_6_]]#0) <{axis = 0 : i32}> ({
// CHECK:           ^bb0([[VAR_arg6_1_:%.+]]: f32, [[VAR_arg7_1_:%.+]]: f32):
// CHECK:             [[VAR_23_1_:%.+]] = arith.addf [[VAR_arg6_1_]], [[VAR_arg7_1_]] : f32
// CHECK:             tt.reduce.return [[VAR_23_1_]] : f32
// CHECK:           }) : (tensor<256x256xf32>) -> tensor<256xf32>
// CHECK:           [[VAR_8_:%.+]] = "tt.reduce"([[VAR_6_]]#1) <{axis = 0 : i32}> ({
// CHECK:           ^bb0([[VAR_arg6_1_:%.+]]: f32, [[VAR_arg7_1_:%.+]]: f32):
// CHECK:             [[VAR_23_2_:%.+]] = arith.addf [[VAR_arg6_1_]], [[VAR_arg7_1_]] : f32
// CHECK:             tt.reduce.return [[VAR_23_2_]] : f32
// CHECK:           }) : (tensor<256x256xf32>) -> tensor<256xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tts.make_tptr [[PARAM_2_]] to sizes: [256], strides: [1], offsets: {{.}}[[VAR_3_]]{{.}}, shape: [0], order: [] : <f32> to tensor<256x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.addi [[VAR_10_]], [[CST_256_]] : index
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:           [[VAR_13_:%.+]] = arith.minsi [[VAR_11_]], [[VAR_12_]] : index
// CHECK:           [[VAR_14_:%.+]] = arith.maxsi [[VAR_13_]], [[VAR_10_]] : index
// CHECK:           [[VAR_15_:%.+]] = arith.subi [[VAR_14_]], [[VAR_10_]] : index
// CHECK:           "tts.store"([[VAR_9_]], [[VAR_7_]], [[VAR_15_]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<256x!tt.ptr<f32>>, tensor<256xf32>, index) -> ()
// CHECK-DAG:       [[VAR_16_:%.+]] = tts.make_tptr [[PARAM_3_]] to sizes: [256], strides: [1], offsets: {{.}}[[VAR_2_]]{{.}}, shape: [0], order: [] : <f32> to tensor<256x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.addi [[VAR_17_]], [[CST_256_]] : index
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:           [[VAR_20_:%.+]] = arith.minsi [[VAR_18_]], [[VAR_19_]] : index
// CHECK:           [[VAR_21_:%.+]] = arith.maxsi [[VAR_20_]], [[VAR_17_]] : index
// CHECK:           [[VAR_22_:%.+]] = arith.subi [[VAR_21_]], [[VAR_17_]] : index
// CHECK:           "tts.store"([[VAR_16_]], [[VAR_8_]], [[VAR_22_]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<256x!tt.ptr<f32>>, tensor<256xf32>, index) -> ()
// CHECK:           tt.return
// CHECK:         }
