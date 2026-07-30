// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  )
  {
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %ldptr = tt.addptr %0, %2 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %stptr = tt.addptr %1, %2 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %c7_i32 = arith.constant 7 : i32
    %splat_c7_i32 = tt.splat %c7_i32 : i32 -> tensor<128xi32>
    %splat_c7_bf16 = arith.sitofp %splat_c7_i32 : tensor<128xi32> to tensor<128xbf16>
    %5 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %mask = arith.cmpi slt, %2, %5 : tensor<128xi32>
    %buff = tt.load %ldptr, %mask, %splat_c7_bf16 : tensor<128x!tt.ptr<bf16>>
    tt.store %stptr, %buff, %mask : tensor<128x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: i32) {
// CHECK-DAG:       [[CST_7_dot_000000_:%.+]] = arith.constant 7.000000e+00 : bf16
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [128], strides: [1], offsets: [0], shape: [0], order: [] : <bf16> to tensor<128x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_1_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [128], strides: [1], offsets: [0], shape: [0], order: [] : <bf16> to tensor<128x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_3_:%.+]] = arith.minsi [[VAR_2_]], [[CST_128_]] : index
// CHECK:           [[VAR_4_:%.+]] = arith.maxsi [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_5_:%.+]] = "tts.load"([[VAR_0_]], [[VAR_4_]], [[CST_7_dot_000000_]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<bf16>>, index, bf16) -> tensor<128xbf16>
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_7_:%.+]] = arith.minsi [[VAR_6_]], [[CST_128_]] : index
// CHECK:           [[VAR_8_:%.+]] = arith.maxsi [[VAR_7_]], [[CST_0_]] : index
// CHECK:           "tts.store"([[VAR_1_]], [[VAR_5_]], [[VAR_8_]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<bf16>>, tensor<128xbf16>, index) -> ()
// CHECK:           tt.return
// CHECK:         }
