// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func @kernel(
    %arg0 : !tt.ptr<bf16>,
    %arg1 : i32
  )
  {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32}:tensor<256xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>

    %splat_arg0 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x256x!tt.ptr<bf16>>
    %2 = tt.addptr %splat_arg0, %1 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>

    // 1x256 pointer should have meaningful stride in outer dimension
    %3 = tt.load %2 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<1x256x!tt.ptr<bf16>>

    %4 = tt.splat %arg1 : i32 -> tensor<1x256xi32>
    // 1x256 pointer should have meaningful stride in outer dimension
    %5 = tt.addptr %2, %4 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>
    tt.store %5, %3 : tensor<1x256x!tt.ptr<bf16>>

    %10 = arith.constant 0.0 : bf16
    %11 = tt.splat %10 : bf16 -> tensor<4x256xbf16>

    %c0 = arith.constant 0 : index
    %c12 = arith.constant 12 : index
    %c3 = arith.constant 3 : index
    %i_c3 = arith.constant 3 : i32
    %c256 = arith.constant 256 : i32
    %sum_out, %_ptr = scf.for %i = %c0 to %c12 step %c3 iter_args(%sum_iter = %11, %ptr = %2) -> (tensor<4x256xbf16>, tensor<1x256x!tt.ptr<bf16>>) {
        %bptr = tt.broadcast %ptr : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>

        %20 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
        %i_i32 = arith.index_cast %i : index to i32
        %21 = arith.muli %c256, %i_i32 : i32
        %22 = tt.splat %21 : i32 -> tensor<4xi32>
        %23 = arith.muli %20, %22 : tensor<4xi32>
        %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
        %25 = tt.broadcast %24 : tensor<4x1xi32> -> tensor<4x256xi32>

        // %bptr should have zero stride and %30 should have correct stride
        %30 = tt.addptr %bptr, %25 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
        %31 = tt.load %30 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: tensor<4x256x!tt.ptr<bf16>>
        %32 = arith.addf %sum_iter, %31 : tensor<4x256xbf16>

        %40 = tt.splat %c256 : i32 -> tensor<1x256xi32>
        %41 = tt.addptr %ptr, %40 : tensor<1x256x!tt.ptr<bf16>>, tensor<1x256xi32>

        scf.yield %32, %41 : tensor<4x256xbf16>, tensor<1x256x!tt.ptr<bf16>>
    }

    %31 = tt.make_range {end = 4 : i32, start = 0 : i32}:tensor<4xi32>
    %splat_c256 = tt.splat %c256 : i32 -> tensor<4xi32>
    %32 = arith.muli %31, %splat_c256 : tensor<4xi32>
    %33 = tt.expand_dims %32 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %34 = tt.broadcast %33 : tensor<4x1xi32> -> tensor<4x256xi32>
    %35 = tt.broadcast %2 : tensor<1x256x!tt.ptr<bf16>> -> tensor<4x256x!tt.ptr<bf16>>
    %36 = tt.addptr %35, %34 : tensor<4x256x!tt.ptr<bf16>>, tensor<4x256xi32>
    tt.store %36, %sum_out : tensor<4x256x!tt.ptr<bf16>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : bf16
// CHECK-DAG:       [[CST_256_1_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<4x256xbf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : bf16) outs([[VAR_0_]] : tensor<4x256xbf16>) -> tensor<4x256xbf16>
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1, 256], strides: [256, 1] : memref<*xbf16> to memref<1x256xbf16, strided<[256, 1]>>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<1x256xbf16>
// CHECK:           memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<1x256xbf16, strided<[256, 1]>> to memref<1x256xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<1x256xbf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_1_]] : i32 to index
// CHECK:           [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [1, 256], strides: [256, 1] : memref<*xbf16> to memref<1x256xbf16, strided<[256, 1], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_2_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<1x256xbf16>, memref<1x256xbf16, strided<[256, 1], offset: ?>>) -> ()
// CHECK-DAG:       [[VAR_4_:%.+]]:2 = scf.for [[VAR_arg8_:%.+]] = [[CST_0_]] to [[CST_12_]] step [[CST_3_]] iter_args([[VAR_arg9_:%.+]] = [[VAR_1_]], [[VAR_arg10_:%.+]] = [[CST_0_]]) -> (tensor<4x256xbf16>, index) {
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.index_cast [[VAR_arg8_]] : index to i32
// CHECK:             [[VAR_6_:%.+]] = arith.muli [[VAR_5_]], [[CST_256_]] : i32
// CHECK:             [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : i32 to index
// CHECK-DAG:         [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg10_]]{{.}}, sizes: [4, 256], strides: {{.}}[[VAR_7_]], 1{{.}} : memref<*xbf16> to memref<4x256xbf16, strided<[?, 1], offset: ?>>
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() : memref<4x256xbf16>
// CHECK:             memref.copy [[VAR_reinterpret_cast_2_]], [[RES_1_]] : memref<4x256xbf16, strided<[?, 1], offset: ?>> to memref<4x256xbf16>
// CHECK:             [[VAR_8_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<4x256xbf16>
// CHECK:             [[VAR_9_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_arg9_]], [[VAR_8_]] : tensor<4x256xbf16>, tensor<4x256xbf16>) outs([[VAR_arg9_]] : tensor<4x256xbf16>) {
// CHECK:             ^bb0([[IN_0_:%.+]]: bf16, [[IN_1_:%.+]]: bf16, [[IN_2_:%.+]]: bf16):
// CHECK:               [[VAR_11_:%.+]] = arith.addf [[IN_0_]], [[IN_1_]] : bf16
// CHECK:               linalg.yield [[VAR_11_]] : bf16
// CHECK:             } -> tensor<4x256xbf16>
// CHECK:             [[VAR_10_:%.+]] = arith.addi [[VAR_arg10_]], [[CST_256_1_]] : index
// CHECK:             scf.yield [[VAR_9_]], [[VAR_10_]] : tensor<4x256xbf16>, index
// CHECK:           }
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [4, 256], strides: [256, 1] : memref<*xbf16> to memref<4x256xbf16, strided<[256, 1]>>
// CHECK:           bufferization.materialize_in_destination [[VAR_4_]]#0 in writable [[VAR_reinterpret_cast_1_]] : (tensor<4x256xbf16>, memref<4x256xbf16, strided<[256, 1]>>) -> ()
// CHECK:           return
// CHECK:         }
