// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>,
    %arg3 : i32
  )
  {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    // offset = 0, size = 4, stride = 1
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    // offset = [0,0], size = [4,1], stride = [1,0]
    %2 = tt.broadcast %1 : tensor<4x1xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [1,0]
    %arg3splat = tt.splat %arg3 : i32 -> tensor<4x256xi32>
    %offset3 = arith.addi %2, %arg3splat : tensor<4x256xi32>
    // offset = [%arg3,0], size = [4,256], stride = [1,0]
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32}: tensor<256xi32>
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
    // offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %10 = tt.load %9 : tensor<4x256x!tt.ptr<bf16>>
    %11 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %12 = tt.addptr %11, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg1, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %13 = tt.load %12 : tensor<4x256x!tt.ptr<bf16>>
    %14 = arith.addf %10, %13 : tensor<4x256xbf16>
    %15 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %16 = tt.addptr %15, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg2, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    tt.store %16, %14 : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: !tt.ptr<bf16>, [[PARAM_3_:%.+]]: i32) {
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_1_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [4, 256], strides: [1, [[CST_5_]]{{.}}, offsets: {{.}}[[VAR_0_]], 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tts.load"([[VAR_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_4_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [4, 256], strides: [1, [[CST_5_]]{{.}}, offsets: {{.}}[[VAR_3_]], 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK:           [[VAR_5_:%.+]] = "tts.load"([[VAR_4_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.addf [[VAR_2_]], [[VAR_5_]] : tensor<4x256xbf16>
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_8_:%.+]] = tts.make_tptr [[PARAM_2_]] to sizes: [4, 256], strides: [1, [[CST_5_]]{{.}}, offsets: {{.}}[[VAR_7_]], 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK:           "tts.store"([[VAR_8_]], [[VAR_6_]]) <{static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xbf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
