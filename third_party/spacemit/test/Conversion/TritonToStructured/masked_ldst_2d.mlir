// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32,
  %arg3 : i32
  )
  {
    // Mimic a scenario where the raw pointer points to a buffer with dimension (1024, 1024)
    // in row-major, but the actual tensor size is (arg2, arg3).
    // We are trying to load a 128x256 sub-buffer starting at (2, 3).
    // The resulting memref:
    //  offset = 3074
    //  size[1] = 128
    //  size[0] = 256
    //  stride[0] = 1024
    //  stride[1] = 1
    %0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
    %1 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
    // horizontal index
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c2 = arith.constant 2 : i32
    %c2tensor = tt.splat %c2 : i32 -> tensor<128xi32>
    %offset2 = arith.addi %2, %c2tensor : tensor<128xi32>
    %3 = tt.expand_dims %offset2 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %4 = tt.broadcast %3 : tensor<128x1xi32> -> tensor<128x256xi32>
    // vertical index
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %c3 = arith.constant 3 : i32
    %c3tensor = tt.splat %c3 : i32 -> tensor<256xi32>
    %offset5 = arith.addi %5, %c3tensor : tensor<256xi32>
    %c1024 = arith.constant 1024 : i32
    %c1024tensor = tt.splat %c1024 : i32 -> tensor<256xi32>
    %scale5 = arith.muli %offset5, %c1024tensor : tensor<256xi32>
    %6 = tt.expand_dims %scale5 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %7 = tt.broadcast %6 : tensor<1x256xi32> -> tensor<128x256xi32>
    // combined index
    %index = arith.addi %4, %7 : tensor<128x256xi32>
    %ldptr = tt.addptr %0, %index : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %stptr = tt.addptr %1, %index : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    // other value for masked load
    %cnan = arith.constant 0xFF80 : bf16
    %nans = tt.splat %cnan : bf16 -> tensor<128x256xbf16>
    // horizontal mask
    %8 = tt.splat %arg2 : i32 -> tensor<128xi32>
    %9 = arith.cmpi slt, %offset2, %8 : tensor<128xi32>
    %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
    %11 = tt.broadcast %10 : tensor<128x1xi1> -> tensor<128x256xi1>
    // vertical mask
    %12 = tt.splat %arg3 : i32 -> tensor<256xi32>
    %13 = arith.cmpi slt, %offset5, %12 : tensor<256xi32>
    %14 = tt.expand_dims %13 {axis = 0 : i32} : tensor<256xi1> -> tensor<1x256xi1>
    %15 = tt.broadcast %14 : tensor<1x256xi1> -> tensor<128x256xi1>
    // combined mask
    %mask = arith.andi %11, %15 : tensor<128x256xi1>
    // dim0 = min(%arg2, 128), dim1 = min(%arg3, 256)
    %buff = tt.load %ldptr, %mask, %nans : tensor<128x256x!tt.ptr<bf16>>
    tt.store %stptr, %buff, %mask : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF80 : bf16
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_259_:%.+]] = arith.constant 259 : index
// CHECK-DAG:       [[CST_130_:%.+]] = arith.constant 130 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [128, 256], strides: [1, [[CST_1024_]]{{.}}, offsets: {{.}}[[CST_2_]], 3072{{.}}, shape: [0, 0], order: [] : <bf16> to tensor<128x256x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_1_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [128, 256], strides: [1, [[CST_1024_]]{{.}}, offsets: {{.}}[[CST_2_]], 3072{{.}}, shape: [0, 0], order: [] : <bf16> to tensor<128x256x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_3_:%.+]] = arith.minsi [[VAR_2_]], [[CST_130_]] : index
// CHECK:           [[VAR_4_:%.+]] = arith.maxsi [[VAR_3_]], [[CST_2_]] : index
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.subi [[VAR_4_]], [[CST_2_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_7_:%.+]] = arith.minsi [[VAR_6_]], [[CST_259_]] : index
// CHECK:           [[VAR_8_:%.+]] = arith.maxsi [[VAR_7_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.subi [[VAR_8_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.minsi [[VAR_5_]], [[CST_128_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.minsi [[VAR_9_]], [[CST_256_]] : index
// CHECK-DAG:       [[VAR_12_:%.+]] = "tts.load"([[VAR_0_]], [[VAR_10_]], [[VAR_11_]], [[CST_0_]]) <{operandSegmentSizes = array<i32: 1, 2, 1>, static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808>}> : (tensor<128x256x!tt.ptr<bf16>>, index, index, bf16) -> tensor<128x256xbf16>
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:           [[VAR_14_:%.+]] = arith.minsi [[VAR_13_]], [[CST_130_]] : index
// CHECK:           [[VAR_15_:%.+]] = arith.maxsi [[VAR_14_]], [[CST_2_]] : index
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.subi [[VAR_15_]], [[CST_2_]] : index
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_18_:%.+]] = arith.minsi [[VAR_17_]], [[CST_259_]] : index
// CHECK:           [[VAR_19_:%.+]] = arith.maxsi [[VAR_18_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.subi [[VAR_19_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.minsi [[VAR_16_]], [[CST_128_]] : index
// CHECK:           [[VAR_22_:%.+]] = arith.minsi [[VAR_20_]], [[CST_256_]] : index
// CHECK:           "tts.store"([[VAR_1_]], [[VAR_12_]], [[VAR_21_]], [[VAR_22_]]) <{static_mask_dims = array<i64: -9223372036854775808, -9223372036854775808>}> : (tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xbf16>, index, index) -> ()
// CHECK:           tt.return
// CHECK:         }
