// RUN: triton-opt  '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true'  --split-input-file %s | FileCheck %s

module {
  tt.func public @test_cmp(%arg0: tensor<128xi32>) -> tensor<128xi1> {
    %cst_12 = arith.constant dense<0> : tensor<128xi32>
    %cst_13 = arith.constant dense<1> : tensor<128xi32>
    %cst_14 = arith.constant dense<100> : tensor<128xi32>
    %39 = arith.cmpi slt, %arg0, %cst_14 : tensor<128xi32>
    %40 = arith.select %39, %cst_13, %cst_12 : tensor<128xi1>, tensor<128xi32>
    %41 = arith.cmpi ne, %40, %cst_12 : tensor<128xi32>
    tt.return %41 : tensor<128xi1>
  }
}

// CHECK-LABEL:   tt.func public @test_cmp(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<128xi32>) -> tensor<128xi1> {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0> : tensor<128xi32>
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<1> : tensor<128xi32>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<100> : tensor<128xi32>
// CHECK:           %[[VAL_4:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VAL_3]] : tensor<128xi32>
// CHECK:           %[[VAL_5:.*]] = arith.select %[[VAL_4]], %[[VAL_2]], %[[VAL_1]] : tensor<128xi1>, tensor<128xi32>
// CHECK:           %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_1]] : tensor<128xi32>
// CHECK:           tt.return %[[VAL_6]] : tensor<128xi1>
// CHECK:         }
