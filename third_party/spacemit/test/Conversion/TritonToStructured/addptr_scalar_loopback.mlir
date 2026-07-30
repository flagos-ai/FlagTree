// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

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

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<bf16>, [[PARAM_1_:%.+]]: !tt.ptr<bf16>, [[PARAM_2_:%.+]]: i32) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.addptr [[PARAM_0_]], [[PARAM_2_]] : !tt.ptr<bf16>, i32
// CHECK-DAG:       [[VAR_1_:%.+]] = tt.addptr [[PARAM_1_]], [[PARAM_2_]] : !tt.ptr<bf16>, i32
// CHECK:           [[LOAD_VAR_0_MEM_:%.+]] = tt.load [[VAR_0_]] : !tt.ptr<bf16>
// CHECK:           tt.store [[VAR_1_]], [[LOAD_VAR_0_MEM_]] : !tt.ptr<bf16>
// CHECK:           tt.return
// CHECK:         }
