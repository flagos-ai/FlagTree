// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
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

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_1_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_0_]]{{.}}, sizes: [4, 256], strides: [1, 5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, 5], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<4x256xbf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<4x256xbf16, strided<[1, 5], offset: ?>> to memref<4x256xbf16>
// CHECK-DAG:       [[VAR_1_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4x256xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.addi [[VAR_0_]], [[VAR_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_2_]]{{.}}, sizes: [4, 256], strides: [2, 10] : memref<*xbf16> to memref<4x256xbf16, strided<[2, 10], offset: ?>>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<4x256xbf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_0_]], [[RES_1_]] : memref<4x256xbf16, strided<[2, 10], offset: ?>> to memref<4x256xbf16>
// CHECK:           [[VAR_3_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<4x256xbf16>
// CHECK:           [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_1_]], [[VAR_3_]] : tensor<4x256xbf16>, tensor<4x256xbf16>) outs([[VAR_1_]] : tensor<4x256xbf16>) {
// CHECK:           ^bb0([[IN_0_:%.+]]: bf16, [[IN_1_:%.+]]: bf16, [[IN_2_:%.+]]: bf16):
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[IN_0_]], [[IN_1_]] : bf16
// CHECK:             linalg.yield [[VAR_6_]] : bf16
// CHECK:           } -> tensor<4x256xbf16>
// CHECK:           [[VAR_5_:%.+]] = arith.addi [[VAR_2_]], [[VAR_0_]] : index
// CHECK:           [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_5_]]{{.}}, sizes: [4, 256], strides: [3, 15] : memref<*xbf16> to memref<4x256xbf16, strided<[3, 15], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_4_]] in writable [[VAR_reinterpret_cast_2_]] : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[3, 15], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
