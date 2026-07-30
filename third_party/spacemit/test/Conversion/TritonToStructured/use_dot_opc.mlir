// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>
  )
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %c64 = arith.constant 128 : i32
    %1 = tt.splat %c64 : i32 -> tensor<128xi32>
    %2 = arith.muli %0, %1 : tensor<128xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %4 = tt.broadcast %3 : tensor<128x1xi32> -> tensor<128x64xi32>
    %5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %7 = tt.broadcast %6 : tensor<1x64xi32> -> tensor<128x64xi32>
    %8 = arith.addi %4, %7 : tensor<128x64xi32>
    %10 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %11 = tt.expand_dims %10 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %12 = tt.broadcast %11 : tensor<1x256xi32> -> tensor<64x256xi32>
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %c256 = arith.constant 256 : i32
    %14 = tt.splat %c256 : i32 -> tensor<64xi32>
    %15 = arith.muli %13, %14 : tensor<64xi32>
    %16 = tt.expand_dims %15 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %17 = tt.broadcast %16 : tensor<64x1xi32> -> tensor<64x256xi32>
    %18 = arith.addi %12, %17 : tensor<64x256xi32>
    %20 = tt.splat %c256 : i32 -> tensor<128xi32>
    %21 = arith.muli %0, %20 : tensor<128xi32>
    %22 = tt.expand_dims %21 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %23 = tt.broadcast %22 : tensor<128x1xi32> -> tensor<128x256xi32>
    %24 = tt.expand_dims %10 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %25 = tt.broadcast %24 {axis = 0 : i32} : tensor<1x256xi32> -> tensor<128x256xi32>
    %26 = arith.addi %23, %25 : tensor<128x256xi32>
    %30 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<128x64x!tt.ptr<bf16>>
    %31 = tt.addptr %30, %8 : tensor<128x64x!tt.ptr<bf16>>, tensor<128x64xi32>
    %32 = tt.load %31 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<128x64x!tt.ptr<bf16>>
    %40 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<64x256x!tt.ptr<bf16>>
    %41 = tt.addptr %40, %18 : tensor<64x256x!tt.ptr<bf16>>, tensor<64x256xi32>
    %42 = tt.load %41 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<64x256x!tt.ptr<bf16>>
    %50 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<128x256x!tt.ptr<bf16>>
    %51 = tt.addptr %50, %26 : tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xi32>
    %cf0 = arith.constant 0.0 : bf16
    %71 = tt.splat %cf0 : bf16 -> tensor<128x256xbf16>
    %60 = tt.dot %32, %42, %71 {inputPrecision = 2 : i32, maxNumImpreciseAcc = 0 : i32} : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
    tt.store %51, %60 : tensor<128x256x!tt.ptr<bf16>>
    tt.store %51, %71 : tensor<128x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: !tt.ptr<bf16>) {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<0.000000e+00> : tensor<128x256xbf16>
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK:           [[VAR_0_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [128, 64], strides: {{.}}[[CST_128_]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16> to tensor<128x64x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tts.load"([[VAR_0_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<128x64x!tt.ptr<bf16>>) -> tensor<128x64xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [64, 256], strides: {{.}}[[CST_256_]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16> to tensor<64x256x!tt.ptr<bf16>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = "tts.load"([[VAR_2_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<64x256x!tt.ptr<bf16>>) -> tensor<64x256xbf16>
// CHECK-DAG:       [[VAR_4_:%.+]] = tts.make_tptr [[PARAM_2_]] to sizes: [128, 256], strides: {{.}}[[CST_256_]], 1], offsets: [0, 0], shape: [0, 0], order: [] : <bf16> to tensor<128x256x!tt.ptr<bf16>>
// CHECK:           [[VAR_5_:%.+]] = tt.dot [[VAR_1_]], [[VAR_3_]], [[VAR_cst_]] : tensor<128x64xbf16> * tensor<64x256xbf16> -> tensor<128x256xbf16>
// CHECK:           "tts.store"([[VAR_4_]], [[VAR_5_]]) <{static_mask_dims = array<i64>}> : (tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xbf16>) -> ()
// CHECK:           "tts.store"([[VAR_4_]], [[VAR_cst_]]) <{static_mask_dims = array<i64>}> : (tensor<128x256x!tt.ptr<bf16>>, tensor<128x256xbf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
