// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : !tt.ptr<bf16>,
    %arg2 : !tt.ptr<bf16>,
    %arg3 : i32,
    %arg4 : i32
  )
  {
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
    // offset = 0, size = 4, stride = 1
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    // offset = [0,0], size = [4,1], stride = [1,0]
    %2 = tt.broadcast %1 : tensor<4x1xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [1,0]
    %arg3splat = tt.splat %arg3 : i32 -> tensor<4x256xi32>
    %offset3 = arith.addi %2, %arg3splat : tensor<4x256xi32>
    // offset = [%arg3,0], size = [4,256], stride = [1,0]
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    // offset = 0, size = 256, stride = 1
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    // offset = [0,0], size = [1,256], stride = [0,1]
    %5 = tt.broadcast %4 : tensor<1x256xi32> -> tensor<4x256xi32>
    // offset = [0,0], size = [4,256], stride = [0,1]
    %c5 = arith.constant 5 : i32
    %splat6 = tt.splat %c5 : i32 -> tensor<4x256xi32>
    // scalar = 5
    %scale5 = arith.muli %5, %splat6 : tensor<4x256xi32> // Why we never called the conversion function for the inputs here?
    // offset = [0,0], size = [4,256], stride = [0,5]
    %7 = arith.addi %offset3, %scale5: tensor<4x256xi32> // Why we never called the conversion function for the inputs here?
    // offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %8 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>> // Why is the input unknown
    %9 = tt.addptr %8, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg0, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %19 = tt.load %9 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>> // this will be replaced with a memref.copy
    %11 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %12 = tt.addptr %11, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg1, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %sum_out, %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%sum_iter = %19, %ptr_iter = %12) -> (tensor<4x256xbf16>, tensor<4x256x!tt.ptr<bf16>>) {
        %20 = tt.load %ptr_iter {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
        %sum = arith.addf %sum_iter, %20 : tensor<4x256xbf16>
        // pointer updates
        %17 = tt.splat %i_c3 : i32 -> tensor<4x256xi32>
        // offset: [3, 0], size = [4, 256], stride [0, 0]
        %ptr = tt.addptr %ptr_iter, %17 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
        // source: %arg1, offset = [%arg3+%i, 0], size = [4, 256], stride = [1, 5]
        scf.yield %sum, %ptr : tensor<4x256xbf16>, tensor<4x256x!tt.ptr<bf16>>
    }
    %15 = tt.splat %arg2 : !tt.ptr<bf16> -> tensor<4x256x!tt.ptr<bf16>>
    %16 = tt.addptr %15, %7 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    // source: %arg2, offset = [%arg3, 0], size = [4, 256], stride = [1, 5]
    tt.store %16, %sum_out : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: memref<*xbf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_0_]]{{.}}, sizes: [4, 256], strides: [1, 5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, 5], offset: ?>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<4x256xbf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<4x256xbf16, strided<[1, 5], offset: ?>> to memref<4x256xbf16>
// CHECK:           [[VAR_1_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4x256xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]]:2 = scf.for [[VAR_arg11_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg12_:%.+]] = [[VAR_1_]], [[VAR_arg13_:%.+]] = [[VAR_0_]]) -> (tensor<4x256xbf16>, index) {
// CHECK-DAG:         [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg13_]]{{.}}, sizes: [4, 256], strides: [1, 5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, 5], offset: ?>>
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() : memref<4x256xbf16>
// CHECK:             memref.copy [[VAR_reinterpret_cast_1_]], [[RES_1_]] : memref<4x256xbf16, strided<[1, 5], offset: ?>> to memref<4x256xbf16>
// CHECK:             [[VAR_3_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<4x256xbf16>
// CHECK:             [[VAR_4_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_arg12_]], [[VAR_3_]] : tensor<4x256xbf16>, tensor<4x256xbf16>) outs([[VAR_arg12_]] : tensor<4x256xbf16>) {
// CHECK:             ^bb0([[IN_0_:%.+]]: bf16, [[IN_1_:%.+]]: bf16, [[IN_2_:%.+]]: bf16):
// CHECK:               [[VAR_6_:%.+]] = arith.addf [[IN_0_]], [[IN_1_]] : bf16
// CHECK:               linalg.yield [[VAR_6_]] : bf16
// CHECK:             } -> tensor<4x256xbf16>
// CHECK:             [[VAR_5_:%.+]] = arith.addi [[VAR_arg13_]], [[CST_3_]] : index
// CHECK:             scf.yield [[VAR_4_]], [[VAR_5_]] : tensor<4x256xbf16>, index
// CHECK:           }
// CHECK:           [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: {{.}}[[VAR_0_]]{{.}}, sizes: [4, 256], strides: [1, 5] : memref<*xbf16> to memref<4x256xbf16, strided<[1, 5], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_2_]]#0 in writable [[VAR_reinterpret_cast_0_]] : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[1, 5], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
