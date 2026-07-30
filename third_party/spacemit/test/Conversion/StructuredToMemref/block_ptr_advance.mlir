// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @matmul_kernel_with_block_pointers_01234567891011(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : bf16
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = arith.extsi %arg7 : i32 to i64
    %4 = tt.make_tensor_ptr %arg0, [%0, %1], [%2, %3], [%arg12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    %5 = tt.advance %4, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
    %6 = tt.splat %cst : bf16 -> tensor<128x64xbf16>
    %7:3 = scf.for %arg14 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg15 = %6, %arg16 = %5, %arg17 = %4) -> (tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>)  : i32 {
      %13 = tt.load %arg16 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>>
      %14 = tt.load %arg17 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>>
      %15 = arith.addf %13, %14 : tensor<128x64xbf16>
      %16 = arith.addf %arg15, %15 : tensor<128x64xbf16>
      %17 = tt.advance %arg16, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
      %18 = tt.advance %arg17, [%c64_i32, %c0_i32] : <tensor<128x64xbf16>>
      scf.yield %16, %17, %18 : tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>
    }
    %8 = arith.extsi %arg10 : i32 to i64
    %9 = arith.extsi %arg11 : i32 to i64
    %10 = arith.extsi %arg4 : i32 to i64
    %11 = arith.muli %arg13, %c256_i32 : i32
    %12 = tt.make_tensor_ptr %arg2, [%0, %10], [%8, %9], [%arg12, %11] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    tt.store %12, %7#0 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
    tt.return
  }
}

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:  func.func @matmul_kernel_with_block_pointers_01234567891011
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: memref<*xbf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32, [[PARAM_14_:%.+]]: i32, [[PARAM_15_:%.+]]: i32, [[PARAM_16_:%.+]]: i32, [[PARAM_17_:%.+]]: i32, [[PARAM_18_:%.+]]: i32, [[PARAM_19_:%.+]]: i32) {
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i32
// CHECK-DAG:       [[CST_64_1_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : bf16
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<128x64xbf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : bf16) outs([[VAR_0_]] : tensor<128x64xbf16>) -> tensor<128x64xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[PARAM_6_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[PARAM_12_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.muli [[VAR_3_]], [[VAR_2_]] : index
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[PARAM_7_]] : i32 to index
// CHECK:           [[VAR_6_:%.+]] = arith.muli [[VAR_5_]], [[CST_64_1_]] : index
// CHECK-DAG:       [[VAR_7_:%.+]]:3 = scf.for [[VAR_arg20_:%.+]] = [[CST_0_]] to [[PARAM_5_]] step [[CST_64_]] iter_args([[VAR_arg21_:%.+]] = [[VAR_1_]], [[VAR_arg22_:%.+]] = [[VAR_6_]], [[VAR_arg23_:%.+]] = [[VAR_4_]]) -> (tensor<128x64xbf16>, index, index)  : i32 {
// CHECK-DAG:         [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_arg23_]]{{.}}, sizes: [128, 64], strides: {{.}}[[VAR_2_]], [[VAR_5_]]{{.}} : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.addi [[VAR_4_]], [[VAR_arg22_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_15_]]{{.}}, sizes: [128, 64], strides: {{.}}[[VAR_2_]], [[VAR_5_]]{{.}} : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<128x64xbf16>
// CHECK:             memref.copy [[VAR_reinterpret_cast_1_]], [[RES_]] : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<128x64xbf16>
// CHECK-DAG:         [[VAR_16_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<128x64xbf16>
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() : memref<128x64xbf16>
// CHECK:             memref.copy [[VAR_reinterpret_cast_0_]], [[RES_1_]] : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<128x64xbf16>
// CHECK:             [[VAR_17_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<128x64xbf16>
// CHECK:             [[VAR_18_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_16_]], [[VAR_17_]] : tensor<128x64xbf16>, tensor<128x64xbf16>) outs([[VAR_16_]] : tensor<128x64xbf16>) {
// CHECK:             ^bb0([[IN_0_:%.+]]: bf16, [[IN_1_:%.+]]: bf16, [[IN_2_:%.+]]: bf16):
// CHECK:               [[VAR_23_:%.+]] = arith.addf [[IN_0_]], [[IN_1_]] : bf16
// CHECK:               linalg.yield [[VAR_23_]] : bf16
// CHECK:             } -> tensor<128x64xbf16>
// CHECK:             [[VAR_19_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_arg21_]], [[VAR_18_]] : tensor<128x64xbf16>, tensor<128x64xbf16>) outs([[VAR_arg21_]] : tensor<128x64xbf16>) {
// CHECK:             ^bb0([[IN_3_:%.+]]: bf16, [[IN_4_:%.+]]: bf16, [[IN_5_:%.+]]: bf16):
// CHECK:               [[VAR_23_1_:%.+]] = arith.addf [[IN_3_]], [[IN_4_]] : bf16
// CHECK:               linalg.yield [[VAR_23_1_]] : bf16
// CHECK:             } -> tensor<128x64xbf16>
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.addi [[VAR_6_]], [[VAR_arg22_]] : index
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.muli [[VAR_2_]], [[CST_64_1_]] : index
// CHECK:             [[VAR_22_:%.+]] = arith.addi [[VAR_21_]], [[VAR_arg23_]] : index
// CHECK:             scf.yield [[VAR_19_]], [[VAR_20_]], [[VAR_22_]] : tensor<128x64xbf16>, index, index
// CHECK:           }
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.muli [[PARAM_13_]], [[CST_256_]] : i32
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.index_cast [[PARAM_10_]] : i32 to index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.muli [[VAR_3_]], [[VAR_9_]] : index
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[PARAM_11_]] : i32 to index
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.index_cast [[VAR_8_]] : i32 to index
// CHECK:           [[VAR_13_:%.+]] = arith.muli [[VAR_12_]], [[VAR_11_]] : index
// CHECK:           [[VAR_14_:%.+]] = arith.addi [[VAR_10_]], [[VAR_13_]] : index
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_2_]] to offset: {{.}}[[VAR_14_]]{{.}}, sizes: [128, 64], strides: {{.}}[[VAR_9_]], [[VAR_11_]]{{.}} : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:           bufferization.materialize_in_destination [[VAR_7_]]#0 in writable [[VAR_reinterpret_cast_]] : (tensor<128x64xbf16>, memref<128x64xbf16, strided<[?, ?], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }
