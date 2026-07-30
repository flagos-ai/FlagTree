// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func public @matmul_kernel_with_block_pointers_01234567891011(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : bf16
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = arith.extsi %arg7 : i32 to i64
    %4 = tt.make_tensor_ptr %arg0, [%0, %1], [%2, %3], [%arg12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    // base: arg0, shape: [arg3, arg5], strides: [%arg6, %arg7], offsets: [%arg12, 0], order: [1, 0]
    %5 = tt.advance %4, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
    // base: arg0, shape: [arg3, arg5], strides: [%arg6, %arg7], offsets: [%arg12, 64], order: [1, 0]
    %6 = tt.splat %cst : bf16 -> tensor<128x64xbf16>
    %7:3 = scf.for %arg14 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg15 = %6, %arg16 = %5, %arg17 = %4) -> (tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>)  : i32 {
      %13 = tt.load %arg16 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>>
      %14 = tt.load %arg17 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>>
      %15 = arith.addf %13, %14 : tensor<128x64xbf16>
      %16 = arith.addf %arg15, %15 : tensor<128x64xbf16>
      %17 = tt.advance %arg16, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
      // offset += [0, 64];
      %18 = tt.advance %arg17, [%c64_i32, %c0_i32] : <tensor<128x64xbf16>>
      // offset += [64, 0];
      scf.yield %16, %17, %18 : tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>
    }
    %8 = arith.extsi %arg10 : i32 to i64
    %9 = arith.extsi %arg11 : i32 to i64
    %10 = arith.extsi %arg4 : i32 to i64
    %11 = arith.muli %arg13, %c256_i32 : i32
    %12 = tt.make_tensor_ptr %arg2, [%0, %10], [%8, %9], [%arg12, %11] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    tt.store %12, %7#0 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
    tt.return
  }
}

// CHECK:         tt.func public @matmul_kernel_with_block_pointers_01234567891011([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: !tt.ptr<bf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : tensor<128x64xbf16>
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_64_1_:%.+]] = arith.constant 64 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_12_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[VAR_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[PARAM_5_]] : i32 to index
// CHECK:           [[VAR_6_:%.+]] = arith.muli [[VAR_4_]], [[CST_64_]] : index
// CHECK-DAG:       [[VAR_7_:%.+]]:3 = scf.for [[VAR_arg14_:%.+]] = [[CST_0_1_]] to [[PARAM_5_]] step [[CST_64_1_]] iter_args([[VAR_arg15_:%.+]] = [[VAR_cst_]], [[VAR_arg16_:%.+]] = [[VAR_6_]], [[VAR_arg17_:%.+]] = [[VAR_2_]]) -> (tensor<128x64xbf16>, index, index)  : i32 {
// CHECK-DAG:         [[VAR_18_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [128, 64], strides: {{.}}[[VAR_0_]], [[VAR_4_]]{{.}}, offsets: {{.}}[[VAR_arg17_]], [[CST_0_]]{{.}}, shape: {{.}}[[VAR_3_]], [[VAR_5_]]{{.}}, order: [1, 0] : <bf16> to !tt.ptr<tensor<128x64xbf16>>
// CHECK-DAG:         [[VAR_19_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [128, 64], strides: {{.}}[[VAR_0_]], [[VAR_4_]]{{.}}, offsets: {{.}}[[VAR_2_]], [[VAR_arg16_]]{{.}}, shape: {{.}}[[VAR_3_]], [[VAR_5_]]{{.}}, order: [1, 0] : <bf16> to !tt.ptr<tensor<128x64xbf16>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_20_:%.+]] = "tts.load"([[VAR_19_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<128x64xbf16>>) -> tensor<128x64xbf16>
// CHECK-DAG:         [[VAR_21_:%.+]] = "tts.load"([[VAR_18_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (!tt.ptr<tensor<128x64xbf16>>) -> tensor<128x64xbf16>
// CHECK:             [[VAR_22_:%.+]] = arith.addf [[VAR_20_]], [[VAR_21_]] : tensor<128x64xbf16>
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.addf [[VAR_arg15_]], [[VAR_22_]] : tensor<128x64xbf16>
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.muli [[VAR_4_]], [[CST_64_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.addi [[VAR_24_]], [[VAR_arg16_]] : index
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.muli [[VAR_0_]], [[CST_64_]] : index
// CHECK:             [[VAR_27_:%.+]] = arith.addi [[VAR_26_]], [[VAR_arg17_]] : index
// CHECK:             scf.yield [[VAR_23_]], [[VAR_25_]], [[VAR_27_]] : tensor<128x64xbf16>, index, index
// CHECK:           }
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.muli [[PARAM_13_]], [[CST_256_]] : i32
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.index_cast [[PARAM_10_]] : i32 to index
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.index_cast [[PARAM_12_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.muli [[VAR_10_]], [[VAR_9_]] : index
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.index_cast [[PARAM_11_]] : i32 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.index_cast [[VAR_8_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.muli [[VAR_14_]], [[VAR_13_]] : index
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:           [[VAR_17_:%.+]] = tts.make_tptr [[PARAM_2_]] to sizes: [128, 64], strides: {{.}}[[VAR_9_]], [[VAR_13_]]{{.}}, offsets: {{.}}[[VAR_11_]], [[VAR_15_]]{{.}}, shape: {{.}}[[VAR_12_]], [[VAR_16_]]{{.}}, order: [1, 0] : <bf16> to !tt.ptr<tensor<128x64xbf16>>
// CHECK:           "tts.store"([[VAR_17_]], [[VAR_7_]]#0) <{static_mask_dims = array<i64>}> : (!tt.ptr<tensor<128x64xbf16>>, tensor<128x64xbf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
