// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @addptr(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
    %1 = tt.addptr %arg1, %c1_i32 : !tt.ptr<f32>, i32
    scf.for %arg2 = %c0_i32 to %c10_i32 step %c2_i32  : i32 {
      %2 = tt.addptr %0, %arg2 : !tt.ptr<f32>, i32
      %3 = tt.addptr %2, %c1_i32 : !tt.ptr<f32>, i32
      %4 = tt.addptr %1, %arg2 : !tt.ptr<f32>, i32
      %5 = tt.addptr %4, %c1_i32 : !tt.ptr<f32>, i32
      %6 = tt.load %2 : !tt.ptr<f32>
      %7 = tt.load %3 : !tt.ptr<f32>
      tt.store %4, %6 : !tt.ptr<f32>
      tt.store %5, %7 : !tt.ptr<f32>
    }
    tt.return
  }
}

// CHECK-LABEL:  func.func @addptr
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK:           scf.for [[I_0_:%.+]] = [[CST_0_]] to [[CST_10_]] step [[CST_2_]]  : i32 {
// CHECK-DAG:         [[VAR_0_:%.+]] = arith.addi [[I_0_]], [[CST_1_]] : i32
// CHECK-DAG:         [[VAR_1_:%.+]] = arith.addi [[I_0_]], [[CST_2_]] : i32
// CHECK:             [[VAR_2_:%.+]] = arith.index_cast [[VAR_0_]] : i32 to index
// CHECK:             [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_2_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = affine.load [[VAR_reinterpret_cast_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK:             [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_4_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_0_MEM_:%.+]] = affine.load [[VAR_reinterpret_cast_0_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_2_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             affine.store [[LOAD_VAR_reinterpret_cast_MEM_]], [[VAR_reinterpret_cast_1_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:             [[VAR_reinterpret_cast_2_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_4_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             affine.store [[LOAD_VAR_reinterpret_cast_0_MEM_]], [[VAR_reinterpret_cast_2_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           return
// CHECK:         }
