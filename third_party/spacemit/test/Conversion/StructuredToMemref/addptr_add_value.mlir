// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32,
  %arg3 : i32
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
  %arg3splat = tt.splat %arg3 : i32 -> tensor<4x256xi32>
  %offset3 = arith.addi %offset2, %arg3splat : tensor<4x256xi32>
  // offset = [%arg2+%arg3,0], size = [4,256], stride = [1,0]
  %c10 = arith.constant 10 : i32
  %c10splat = tt.splat %c10 : i32 -> tensor<4x256xi32>
  %offset4 = arith.addi %offset3, %c10splat : tensor<4x256xi32>
  // offset = [%arg2+%arg3+10,0], size = [4,256], stride = [1,0]
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
  %7 = arith.addi %offset4, %scale5: tensor<4x256xi32>
  // offset = [%arg2+%arg3+10, 0], size = [4, 256], stride = [1, 6]
  %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
  %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>,tensor<4x256xi32>
  // source = %arg0, offset = [%arg2+%arg3+10, 0], size = [4, 256], stride = [1, 6]
  %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
  %11 = tt.addptr %10, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
  // source = %arg1, offset = [%arg2+%arg3+10, 0], size = [4, 256], stride = [1, 6]
  %12 = tt.load %9 : tensor<4x256x!tt.ptr<bf16>>
  tt.store %11, %12 : tensor<4x256x!tt.ptr<bf16>>
  tt.return
  }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32) {
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_2_:%.+]] = arith.addi [[VAR_0_]], [[VAR_1_]] : index
// CHECK:           [[VAR_3_:%.+]] = arith.addi [[VAR_2_]], [[CST_10_]] : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [4, 256], strides: [1, 6] : memref<*xbf16> to memref<4x256xbf16, strided<[1, 6], offset: ?>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [4, 256], strides: [1, 6] : memref<*xbf16> to memref<4x256xbf16, strided<[1, 6], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<4x256xbf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<4x256xbf16, strided<[1, 6], offset: ?>> to memref<4x256xbf16>
// CHECK:           [[VAR_4_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4x256xbf16>
// CHECK:           bufferization.materialize_in_destination [[VAR_4_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[1, 6], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
