// RUN: triton-opt  '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true'  --split-input-file %s | FileCheck %s
module {
  tt.func public @test_splat_cmp(%arg0: i32, %arg1: i32) -> tensor<128xi1> {
    %0 = tt.splat %arg0 : i32 -> tensor<128xi32>
    %1 = tt.splat %arg1 : i32 -> tensor<128xi32>
    %2 = arith.cmpi slt, %0, %1 : tensor<128xi32>
    tt.return %2 : tensor<128xi1>
  }
}

// CHECK-LABEL:   tt.func public @test_splat_cmp(
// CHECK-SAME:                             %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) -> tensor<128xi1> {
// CHECK:           %[[VAL_2:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = tt.splat %[[VAL_2]] : i1 -> tensor<128xi1>
// CHECK:           tt.return %[[VAL_3]] : tensor<128xi1>
// CHECK:         }
