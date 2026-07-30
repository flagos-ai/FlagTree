// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(
    %a : !tt.ptr<i32>,
    %b : !tt.ptr<i32>
  ) -> () {
        // offset calculations
        %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
        // a pointer
        %8 = tt.splat %a : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
        %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
        // b pointer
        %18 = tt.splat %b : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
        %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
        %am = tt.load %9 : tensor<1024x!tt.ptr<i32>>
        %bm = tt.load %19 : tensor<1024x!tt.ptr<i32>>
        %5 = arith.addi %am, %bm : tensor<1024xi32>
        tt.store %19, %5 : tensor<1024x!tt.ptr<i32>>
        tt.return
    }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<i32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<1024x!tt.ptr<i32>>
// CHECK-DAG:       [[VAR_1_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<1024x!tt.ptr<i32>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = "tts.load"([[VAR_0_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<i32>>) -> tensor<1024xi32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tts.load"([[VAR_1_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<i32>>) -> tensor<1024xi32>
// CHECK:           [[VAR_4_:%.+]] = arith.addi [[VAR_2_]], [[VAR_3_]] : tensor<1024xi32>
// CHECK:           "tts.store"([[VAR_1_]], [[VAR_4_]]) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>) -> ()
// CHECK:           tt.return
// CHECK:         }
