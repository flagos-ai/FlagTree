// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func public @matmul_kernel_0123456789101112131415(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) {
    %c63_i32 = arith.constant 63 : i32
    %c255_i32 = arith.constant 255 : i32
    %c127_i32 = arith.constant 127 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg3, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg4, %c255_i32 : i32
    %4 = arith.divsi %3, %c256_i32 : i32
    %5 = arith.addi %arg5, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.muli %4, %c8_i32 : i32
    %8 = arith.divsi %0, %7 : i32
    %9 = arith.muli %8, %c8_i32 : i32
    %10 = arith.subi %2, %9 : i32
    %11 = arith.cmpi slt, %10, %c8_i32 : i32
    %12 = arith.select %11, %10, %c8_i32 : i32
    %13 = arith.remsi %0, %12 : i32
    %14 = arith.addi %9, %13 : i32
    %15 = arith.remsi %0, %7 : i32
    %16 = arith.divsi %15, %12 : i32
    %17 = arith.muli %14, %c128_i32 : i32
    %18 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %19 = tt.splat %17 : i32 -> tensor<128xi32>
    %20 = arith.addi %19, %18 : tensor<128xi32>
    %21 = arith.muli %16, %c256_i32 : i32
    %22 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %23 = tt.splat %21 : i32 -> tensor<256xi32>
    %24 = arith.addi %23, %22 : tensor<256xi32>
    %25 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %26 = tt.expand_dims %20 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %27 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %28 = arith.muli %26, %27 : tensor<128x1xi32>
    %29 = tt.expand_dims %25 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %30 = tt.splat %arg7 : i32 -> tensor<1x64xi32>
    %31 = arith.muli %29, %30 : tensor<1x64xi32>
    %32 = tt.broadcast %28 : tensor<128x1xi32> -> tensor<128x64xi32>
    %33 = tt.broadcast %31 : tensor<1x64xi32> -> tensor<128x64xi32>
    %34 = arith.addi %32, %33 : tensor<128x64xi32>
    %35 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    %36 = tt.addptr %35, %34 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    %37 = tt.expand_dims %25 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %38 = tt.splat %arg8 : i32 -> tensor<64x1xi32>
    %39 = arith.muli %37, %38 : tensor<64x1xi32>
    %40 = tt.expand_dims %24 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %41 = tt.splat %arg9 : i32 -> tensor<1x256xi32>
    %42 = arith.muli %40, %41 : tensor<1x256xi32>
    %43 = tt.broadcast %39 : tensor<64x1xi32> -> tensor<64x256xi32>
    %44 = tt.broadcast %42 : tensor<1x256xi32> -> tensor<64x256xi32>
    %45 = arith.addi %43, %44 : tensor<64x256xi32>
    %46 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x256x!tt.ptr<bf16>>
    %47 = tt.addptr %46, %45 : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>
    %48 = tt.splat %cst : f32 -> tensor<128x256xf32>
    %49 = arith.muli %arg7, %c64_i32 : i32
    %50 = tt.splat %49 : i32 -> tensor<128x64xi32>
    %51 = arith.muli %arg8, %c64_i32 : i32
    %52 = tt.splat %51 : i32 -> tensor<64x256xi32>
    %53:3 = scf.for %arg12 = %c0_i32 to %6 step %c1_i32 iter_args(%arg13 = %48, %arg14 = %36, %arg15 = %47) -> (tensor<128x256xf32>, tensor<128x64x!tt.ptr<bf16>>, tensor<64x256x!tt.ptr<bf16>>)  : i32 {
      %71 = tt.load %arg14 : tensor<128x64x!tt.ptr<bf16>>
      %72 = tt.load %arg15 : tensor<64x256x!tt.ptr<bf16>>
      %73 = tt.dot %71, %72, %48 {inputPrecision = 0 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xf32>
      %74 = arith.addf %arg13, %73 : tensor<128x256xf32>
      %75 = tt.addptr %arg14, %50 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
      %76 = tt.addptr %arg15, %52 : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>
      scf.yield %74, %75, %76 : tensor<128x256xf32>, tensor<128x64x!tt.ptr<bf16>>, tensor<64x256x!tt.ptr<bf16>>
    }
    %54 = arith.truncf %53#0 : tensor<128x256xf32> to tensor<128x256xbf16>
    %55 = tt.splat %arg10 : i32 -> tensor<128x1xi32>
    %56 = arith.muli %55, %26 : tensor<128x1xi32>
    %57 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x1x!tt.ptr<bf16>>
    %58 = tt.addptr %57, %56 : tensor<128x1x!tt.ptr<bf16>>, tensor<128x1xi32>
    %59 = tt.splat %arg11 : i32 -> tensor<1x256xi32>
    %60 = arith.muli %59, %40 : tensor<1x256xi32>
    %61 = tt.broadcast %58 : tensor<128x1x!tt.ptr<bf16>> -> tensor<128x256x!tt.ptr<bf16>>
    %62 = tt.broadcast %60 : tensor<1x256xi32> -> tensor<128x256xi32>
    %63 = tt.addptr %61, %62 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %64 = tt.splat %arg3 : i32 -> tensor<128x1xi32>
    %65 = arith.cmpi slt, %26, %64 : tensor<128x1xi32>
    %66 = tt.splat %arg4 : i32 -> tensor<1x256xi32>
    %67 = arith.cmpi slt, %40, %66 : tensor<1x256xi32>
    %68 = tt.broadcast %65 : tensor<128x1xi1> -> tensor<128x256xi1>
    %69 = tt.broadcast %67 : tensor<1x256xi1> -> tensor<128x256xi1>
    %70 = arith.andi %68, %69 : tensor<128x256xi1>
    tt.store %63, %54, %70 : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func public @matmul_kernel_0123456789101112131415([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: !tt.ptr<bf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32) {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : tensor<128x256xf32>
// CHECK-DAG:       [[CST_63_:%.+]] = arith.constant 63 : i32
// CHECK-DAG:       [[CST_255_:%.+]] = arith.constant 255 : i32
// CHECK-DAG:       [[CST_127_:%.+]] = arith.constant 127 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i32
// CHECK-DAG:       [[CST_256_1_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_128_1_:%.+]] = arith.constant 128 : i32
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.addi [[PARAM_3_]], [[CST_127_]] : i32
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.divsi [[VAR_1_]], [[CST_128_1_]] : i32
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.addi [[PARAM_4_]], [[CST_255_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.divsi [[VAR_3_]], [[CST_256_1_]] : i32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.addi [[PARAM_5_]], [[CST_63_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.divsi [[VAR_5_]], [[CST_64_]] : i32
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.muli [[VAR_4_]], [[CST_8_]] : i32
// CHECK:           [[VAR_8_:%.+]] = arith.divsi [[VAR_0_]], [[VAR_7_]] : i32
// CHECK:           [[VAR_9_:%.+]] = arith.muli [[VAR_8_]], [[CST_8_]] : i32
// CHECK:           [[VAR_10_:%.+]] = arith.subi [[VAR_2_]], [[VAR_9_]] : i32
// CHECK:           [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_10_]], [[CST_8_]] : i32
// CHECK:           [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[VAR_10_]], [[CST_8_]] : i32
// CHECK:           [[VAR_13_:%.+]] = arith.remsi [[VAR_0_]], [[VAR_12_]] : i32
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.addi [[VAR_9_]], [[VAR_13_]] : i32
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.remsi [[VAR_0_]], [[VAR_7_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.divsi [[VAR_15_]], [[VAR_12_]] : i32
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.muli [[VAR_14_]], [[CST_128_1_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.index_cast [[VAR_17_]] : i32 to index
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.index_cast [[VAR_17_]] : i32 to index
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.muli [[VAR_16_]], [[CST_256_1_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.index_cast [[VAR_20_]] : i32 to index
// CHECK-DAG:       [[VAR_22_:%.+]] = arith.index_cast [[VAR_20_]] : i32 to index
// CHECK-DAG:       [[VAR_23_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_24_:%.+]] = arith.muli [[VAR_19_]], [[VAR_23_]] : index
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.index_cast [[PARAM_8_]] : i32 to index
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.index_cast [[PARAM_9_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.muli [[VAR_22_]], [[VAR_27_]] : index
// CHECK-DAG:       [[VAR_29_:%.+]] = arith.muli [[PARAM_7_]], [[CST_64_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_30_:%.+]] = arith.index_cast [[VAR_29_]] : i32 to index
// CHECK-DAG:       [[VAR_31_:%.+]] = arith.muli [[PARAM_8_]], [[CST_64_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_32_:%.+]] = arith.index_cast [[VAR_31_]] : i32 to index
// CHECK-DAG:       [[VAR_33_:%.+]]:3 = scf.for [[VAR_arg12_:%.+]] = [[CST_0_1_]] to [[VAR_6_]] step [[CST_1_]] iter_args([[VAR_arg13_:%.+]] = [[VAR_cst_]], [[VAR_arg14_:%.+]] = [[VAR_24_]], [[VAR_arg15_:%.+]] = [[CST_0_]]) -> (tensor<128x256xf32>, index, index)  : i32 {
// CHECK-DAG:         [[VAR_54_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [64, 256], strides: {{.}}[[VAR_26_]], [[VAR_27_]]{{.}}, offsets: {{.}}[[VAR_arg15_]], [[VAR_28_]]{{.}}, shape: [0, 0], order: [] : <bf16> to tensor<64x256x!tt.ptr<bf16>>
// CHECK-DAG:         [[VAR_55_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [128, 64], strides: {{.}}[[VAR_23_]], [[VAR_25_]]{{.}}, offsets: {{.}}[[VAR_arg14_]], [[CST_0_]]{{.}}, shape: [0, 0], order: [] : <bf16> to tensor<128x64x!tt.ptr<bf16>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_56_:%.+]] = "tts.load"([[VAR_55_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<128x64x!tt.ptr<bf16>>) -> tensor<128x64xbf16>
// CHECK-DAG:         [[VAR_57_:%.+]] = "tts.load"([[VAR_54_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x256x!tt.ptr<bf16>>) -> tensor<64x256xbf16>
// CHECK:             [[VAR_58_:%.+]] = tt.dot [[VAR_56_]], [[VAR_57_]], [[VAR_cst_]], inputPrecision = tf32 : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xf32>
// CHECK-DAG:         [[VAR_59_:%.+]] = arith.addf [[VAR_arg13_]], [[VAR_58_]] : tensor<128x256xf32>
// CHECK-DAG:         [[VAR_60_:%.+]] = arith.addi [[VAR_arg14_]], [[VAR_30_]] : index
// CHECK-DAG:         [[VAR_61_:%.+]] = arith.addi [[VAR_arg15_]], [[VAR_32_]] : index
// CHECK:             scf.yield [[VAR_59_]], [[VAR_60_]], [[VAR_61_]] : tensor<128x256xf32>, index, index
// CHECK:           }
// CHECK-DAG:       [[VAR_34_:%.+]] = arith.truncf [[VAR_33_]]#0 : tensor<128x256xf32> to tensor<128x256xbf16>
// CHECK-DAG:       [[VAR_35_:%.+]] = arith.index_cast [[PARAM_10_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_36_:%.+]] = arith.muli [[VAR_18_]], [[VAR_35_]] : index
// CHECK-DAG:       [[VAR_37_:%.+]] = arith.index_cast [[PARAM_11_]] : i32 to index
// CHECK:           [[VAR_38_:%.+]] = arith.muli [[VAR_21_]], [[VAR_37_]] : index
// CHECK-DAG:       [[VAR_39_:%.+]] = tts.make_tptr [[PARAM_2_]] to sizes: [128, 256], strides: {{.}}[[VAR_35_]], [[VAR_37_]]{{.}}, offsets: {{.}}[[VAR_36_]], [[VAR_38_]]{{.}}, shape: [0, 0], order: [] : <bf16> to tensor<128x256x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_40_:%.+]] = arith.index_cast [[VAR_17_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_41_:%.+]] = arith.addi [[VAR_40_]], [[CST_128_]] : index
// CHECK-DAG:       [[VAR_42_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_43_:%.+]] = arith.minsi [[VAR_41_]], [[VAR_42_]] : index
// CHECK:           [[VAR_44_:%.+]] = arith.maxsi [[VAR_43_]], [[VAR_40_]] : index
// CHECK-DAG:       [[VAR_45_:%.+]] = arith.subi [[VAR_44_]], [[VAR_40_]] : index
// CHECK-DAG:       [[VAR_46_:%.+]] = arith.index_cast [[VAR_20_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_47_:%.+]] = arith.addi [[VAR_46_]], [[CST_256_]] : index
// CHECK-DAG:       [[VAR_48_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:           [[VAR_49_:%.+]] = arith.minsi [[VAR_47_]], [[VAR_48_]] : index
// CHECK:           [[VAR_50_:%.+]] = arith.maxsi [[VAR_49_]], [[VAR_46_]] : index
// CHECK-DAG:       [[VAR_51_:%.+]] = arith.subi [[VAR_50_]], [[VAR_46_]] : index
// CHECK-DAG:       [[VAR_52_:%.+]] = arith.minsi [[VAR_45_]], [[CST_128_]] : index
// CHECK:           [[VAR_53_:%.+]] = arith.minsi [[VAR_51_]], [[CST_256_]] : index
// CHECK:           "tts.store"([[VAR_39_]], [[VAR_34_]], [[VAR_52_]], [[VAR_53_]]) <{static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808>}> : (tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xbf16>, index, index) -> ()
// CHECK:           tt.return
// CHECK:         }
