// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

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
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: !tt.ptr<bf16>, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32, [[PARAM_14_:%.+]]: i32, [[PARAM_15_:%.+]]: i32, [[PARAM_16_:%.+]]: i32, [[PARAM_17_:%.+]]: i32, [[PARAM_18_:%.+]]: i32, [[PARAM_19_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : bf16
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = tensor.empty() : tensor<128x64xbf16>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = linalg.fill ins([[CST_0_dot_000000_]] : bf16) outs([[VAR_0_]] : tensor<128x64xbf16>) -> tensor<128x64xbf16>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.extsi [[PARAM_3_]] : i32 to i64
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.extsi [[PARAM_5_]] : i32 to i64
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.extsi [[PARAM_6_]] : i32 to i64
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.extsi [[PARAM_7_]] : i32 to i64
// CHECK:           [[VAR_6_:%.+]] = tt.make_tensor_ptr [[PARAM_0_]], {{.}}[[VAR_2_]], [[VAR_3_]]{{.}}, {{.}}[[VAR_4_]], [[VAR_5_]]{{.}}, {{.}}[[PARAM_12_]], [[CST_0_]]{{.}} {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
// CHECK:           [[VAR_7_:%.+]] = tt.advance [[VAR_6_]], {{.}}[[CST_0_]], [[CST_64_]]{{.}} : <tensor<128x64xbf16>>
// CHECK-DAG:       [[VAR_8_:%.+]]:3 = scf.for [[VAR_arg20_:%.+]] = [[CST_0_]] to [[PARAM_5_]] step [[CST_64_]] iter_args([[VAR_arg21_:%.+]] = [[VAR_1_]], [[VAR_arg22_:%.+]] = [[VAR_7_]], [[VAR_arg23_:%.+]] = [[VAR_6_]]) -> (tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>)  : i32 {
// CHECK-DAG:         [[LOAD_VAR_arg22_MEM_:%.+]] = tt.load [[VAR_arg22_]] {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<128x64xbf16>>
// CHECK-DAG:         [[LOAD_VAR_arg23_MEM_:%.+]] = tt.load [[VAR_arg23_]] {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<128x64xbf16>>
// CHECK:             [[VAR_16_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[LOAD_VAR_arg22_MEM_]], [[LOAD_VAR_arg23_MEM_]] : tensor<128x64xbf16>, tensor<128x64xbf16>) outs([[LOAD_VAR_arg22_MEM_]] : tensor<128x64xbf16>) {
// CHECK:             ^bb0([[IN_0_:%.+]]: bf16, [[IN_1_:%.+]]: bf16, [[IN_2_:%.+]]: bf16):
// CHECK:               [[VAR_20_:%.+]] = arith.addf [[IN_0_]], [[IN_1_]] : bf16
// CHECK:               linalg.yield [[VAR_20_]] : bf16
// CHECK:             } -> tensor<128x64xbf16>
// CHECK:             [[VAR_17_:%.+]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins([[VAR_arg21_]], [[VAR_16_]] : tensor<128x64xbf16>, tensor<128x64xbf16>) outs([[VAR_arg21_]] : tensor<128x64xbf16>) {
// CHECK:             ^bb0([[IN_3_:%.+]]: bf16, [[IN_4_:%.+]]: bf16, [[IN_5_:%.+]]: bf16):
// CHECK:               [[VAR_20_1_:%.+]] = arith.addf [[IN_3_]], [[IN_4_]] : bf16
// CHECK:               linalg.yield [[VAR_20_1_]] : bf16
// CHECK:             } -> tensor<128x64xbf16>
// CHECK-DAG:         [[VAR_18_:%.+]] = tt.advance [[VAR_arg22_]], {{.}}[[CST_0_]], [[CST_64_]]{{.}} : <tensor<128x64xbf16>>
// CHECK-DAG:         [[VAR_19_:%.+]] = tt.advance [[VAR_arg23_]], {{.}}[[CST_64_]], [[CST_0_]]{{.}} : <tensor<128x64xbf16>>
// CHECK:             scf.yield [[VAR_17_]], [[VAR_18_]], [[VAR_19_]] : tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>
// CHECK:           }
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.extsi [[PARAM_10_]] : i32 to i64
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.extsi [[PARAM_11_]] : i32 to i64
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.extsi [[PARAM_4_]] : i32 to i64
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.muli [[PARAM_13_]], [[CST_256_]] : i32
// CHECK:           [[VAR_13_:%.+]] = tt.make_tensor_ptr [[PARAM_2_]], {{.}}[[VAR_2_]], [[VAR_11_]]{{.}}, {{.}}[[VAR_9_]], [[VAR_10_]]{{.}}, {{.}}[[PARAM_12_]], [[VAR_12_]]{{.}} {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
// CHECK:           tt.store [[VAR_13_]], [[VAR_8_]]#0 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<128x64xbf16>>
// CHECK:           return
// CHECK:         }
