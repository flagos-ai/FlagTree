// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : i32
  )
  {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
    // offset = 0, size = 4, stride = 1
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    // offset = [0,0], size = [4,1], stride = [1,0]
    %2 = tt.broadcast %1 : tensor<4x1xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [1,0]
    %arg1splat = tt.splat %arg1 : i32 -> tensor<4x256xi32>
    %offset3 = arith.addi %2, %arg1splat : tensor<4x256xi32>
    // offset = [%arg1,0], size = [4,256], stride = [1,0]
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    // offset = 0, size = 256, stride = 1
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    // offset = [0,0], size = [1,256], stride = [0,1]
    %5 = tt.broadcast %4 : tensor<1x256xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [0,1]
    %6 = arith.constant 5 : i32
    %splat6 = tt.splat %6 : i32 -> tensor<4x256xi32>
    %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [0,5]
    %7 = arith.addi %offset3, %scale5: tensor<4x256xi32>
    // offset = [%arg1, 0], size = [4, 256], stride = [1, 5]
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg1, 0], size = [4, 256], stride = [1, 5]
    %10 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
    %12 = tt.addptr %9, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg1+%arg1, 0], size = [4, 256], stride = [2, 10]
    %13 = tt.load %12 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
    %14 = arith.addf %10, %13 : tensor<4x256xbf16>
    %16 = tt.addptr %12, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg1+%arg1+%arg1, 0], size = [4, 256], stride = [3, 15]
    tt.store %16, %14 : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: i32) {
// CHECK-DAG:       [[CST_15_:%.+]] = arith.constant 15 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_1_]] : i32 to index
// CHECK:           [[VAR_1_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [4, 256], strides: [1, [[CST_5_]]{{.}}, offsets: {{.}}[[VAR_0_]], 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tts.load"([[VAR_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_1_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = arith.addi [[VAR_0_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [4, 256], strides: [2, [[CST_10_]]{{.}}, offsets: {{.}}[[VAR_4_]], 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK:           [[VAR_6_:%.+]] = "tts.load"([[VAR_5_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.addf [[VAR_2_]], [[VAR_6_]] : tensor<4x256xbf16>
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.index_cast [[PARAM_1_]] : i32 to index
// CHECK:           [[VAR_9_:%.+]] = arith.addi [[VAR_4_]], [[VAR_8_]] : index
// CHECK:           [[VAR_10_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [4, 256], strides: [3, [[CST_15_]]{{.}}, offsets: {{.}}[[VAR_9_]], 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK:           "tts.store"([[VAR_10_]], [[VAR_7_]]) <{static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xbf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
