// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  ) {
    %0 = tt.addptr %arg0, %arg2 : !tt.ptr<bf16>, i32
    %1 = tt.addptr %arg1, %arg2 : !tt.ptr<bf16>, i32
    %10 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: !tt.ptr<bf16>
    tt.store %1, %10 : !tt.ptr<bf16>
    tt.return
  }
}

// CHECK-LABEL:  func.func @kernel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xbf16>, [[PARAM_1_:%.+]]: memref<*xbf16>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_0_]]{{.}}, sizes: [1], strides: [1] : memref<*xbf16> to memref<1xbf16, strided<[1], offset: ?>>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_0_]]{{.}}, sizes: [1], strides: [1] : memref<*xbf16> to memref<1xbf16, strided<[1], offset: ?>>
// CHECK-DAG:           [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = affine.load [[VAR_reinterpret_cast_]][0] : memref<1xbf16, strided<[1], offset: ?>>
// CHECK:           affine.store [[LOAD_VAR_reinterpret_cast_MEM_]], [[VAR_reinterpret_cast_0_]][0] : memref<1xbf16, strided<[1], offset: ?>>
// CHECK:           return
// CHECK:         }
