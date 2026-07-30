// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  )
  {
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
  // offset = 0, size = 4, stride = 1
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  // offset = [0,0], size = [4,1], stride = [1,0]
  %2 = tt.broadcast %1 : tensor<4x1xi32> -> tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [1,0]
  %arg2splat = tt.splat %arg2 : i32 -> tensor<4x256xi32>
  %offset2 = arith.addi %2, %arg2splat : tensor<4x256xi32>
  // offset = [%arg2,0], size = [4,256], stride = [1,0]
  %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
  // offset = 0, size = 256, stride = 1
  %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
  // offset = [0,0], size = [1,256], stride = [0,1]
  %5 = tt.broadcast %4 : tensor<1x256xi32> -> tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [0,1]
  %c6 = arith.constant 6 : i32
  %splat6 = tt.splat %c6 : i32 -> tensor<4x256xi32>
  %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32>
  // offset = [0,0], size = [4,256], stride = [0,6]
  %7 = arith.addi %offset2, %scale5: tensor<4x256xi32>
  // offset = [%arg2, 0], size = [4, 256], stride = [1, 6]
  %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
  %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
  // source: arg0, offset = [%arg2, 0], size = [4, 256], stride = [1, 6]
  %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
  %11 = tt.addptr %10, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
  // source: arg1, offset = [%arg2, 0], size = [4, 256], stride = [1, 6]
  %12 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
  tt.store %11, %12 : tensor<4x256x!tt.ptr<bf16>>
  tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: i32) {
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [4, 256], strides: [1, [[CST_6_]]{{.}}, offsets: {{.}}[[VAR_0_]], 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [4, 256], strides: [1, [[CST_6_]]{{.}}, offsets: {{.}}[[VAR_2_]], 0], shape: [0, 0], order: [] : <bf16> to tensor<4x256x!tt.ptr<bf16>>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tts.load"([[VAR_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>) -> tensor<4x256xbf16>
// CHECK:           "tts.store"([[VAR_3_]], [[VAR_4_]]) <{static_mask_dims = array<i64>}> : (tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xbf16>) -> ()
// CHECK:           tt.return
// CHECK:         }
