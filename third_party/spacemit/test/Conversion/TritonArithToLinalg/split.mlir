// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<256x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<256x!tt.ptr<i32>>, tensor<256xi32>
    %3 = tt.load %2 : tensor<256x!tt.ptr<i32>>
    %4 = tt.reshape %3 allow_reorder : tensor<256xi32> -> tensor<128x2xi32>
    %outLHS, %outRHS = tt.split %4 : tensor<128x2xi32> -> tensor<128xi32>
    %5 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %7 = tt.addptr %6, %5 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.store %7, %outLHS : tensor<128x!tt.ptr<i32>>
    %8 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %9 = tt.addptr %8, %5 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.store %9, %outRHS : tensor<128x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK:   func.func @kernel(%arg0: !tt.ptr<i32> {{.*}}, %arg1: !tt.ptr<i32> {{.*}}, %arg2: !tt.ptr<i32> {{.*}}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
// CHECK:     [[EMPTY256:%.+]] = tensor.empty() : tensor<256xi32>
// CHECK:     [[RANGE256:%.+]] = linalg.generic {{.*}} outs([[EMPTY256]] : tensor<256xi32>) {
// CHECK:       ^bb0(%out: i32):
// CHECK:         [[IDX0:%.+]] = linalg.index 0 : index
// CHECK:         [[I32_0:%.+]] = arith.index_cast [[IDX0]] : index to i32
// CHECK:         linalg.yield [[I32_0]] : i32
// CHECK:     } -> tensor<256xi32>
// CHECK:     [[EMPTY_PTR256:%.+]] = tensor.empty() : tensor<256x!tt.ptr<i32>>
// CHECK:     [[SPLAT_ARG0:%.+]] = linalg.fill ins(%arg0 : !tt.ptr<i32>) outs([[EMPTY_PTR256]] : tensor<256x!tt.ptr<i32>>) -> tensor<256x!tt.ptr<i32>>
// CHECK:     [[ADDPTR256:%.+]] = linalg.generic {{.*}} ins([[SPLAT_ARG0]], [[RANGE256]] : tensor<256x!tt.ptr<i32>>, tensor<256xi32>) outs([[SPLAT_ARG0]] : tensor<256x!tt.ptr<i32>>) {
// CHECK:       ^bb0([[IN_PTR:%.+]]: !tt.ptr<i32>, [[IN_I32:%.+]]: i32, [[OUT_PTR:%.+]]: !tt.ptr<i32>):
// CHECK:         [[NEW_PTR:%.+]] = tt.addptr [[IN_PTR]], [[IN_I32]] : !tt.ptr<i32>, i32
// CHECK:         linalg.yield [[NEW_PTR]] : !tt.ptr<i32>
// CHECK:     } -> tensor<256x!tt.ptr<i32>>
// CHECK:     [[LOADED256:%.+]] = tt.load [[ADDPTR256]] : tensor<256x!tt.ptr<i32>>
// CHECK:     [[RESHAPED:%.+]] = tensor.expand_shape [[LOADED256]]
// CHECK:     [[SLICE_LHS:%.+]] = tensor.extract_slice [[RESHAPED]]{{\[}}0, 0{{\]}} [128, 1] [1, 1] : tensor<128x2xi32> to tensor<128xi32>
// CHECK:     [[SLICE_RHS:%.+]] = tensor.extract_slice [[RESHAPED]]{{\[}}0, 1{{\]}} [128, 1] [1, 1] : tensor<128x2xi32> to tensor<128xi32>
// CHECK:     [[EMPTY128:%.+]] = tensor.empty() : tensor<128xi32>
// CHECK:     [[RANGE128:%.+]] = linalg.generic {{.*}} outs([[EMPTY128]] : tensor<128xi32>) {
// CHECK:       ^bb0(%out: i32):
// CHECK:         [[IDX1:%.+]] = linalg.index 0 : index
// CHECK:         [[I32_1:%.+]] = arith.index_cast [[IDX1]] : index to i32
// CHECK:         linalg.yield [[I32_1]] : i32
// CHECK:     } -> tensor<128xi32>
// CHECK:     [[EMPTY_PTR128_1:%.+]] = tensor.empty() : tensor<128x!tt.ptr<i32>>
// CHECK:     [[SPLAT_ARG1:%.+]] = linalg.fill ins(%arg1 : !tt.ptr<i32>) outs([[EMPTY_PTR128_1]] : tensor<128x!tt.ptr<i32>>) -> tensor<128x!tt.ptr<i32>>
// CHECK:     [[ADDPTR128_1:%.+]] = linalg.generic {{.*}} ins([[SPLAT_ARG1]], [[RANGE128]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>) outs([[SPLAT_ARG1]] : tensor<128x!tt.ptr<i32>>) {
// CHECK:       ^bb0([[IN_PTR:%.+]]: !tt.ptr<i32>, [[IN_I32:%.+]]: i32, [[OUT_PTR:%.+]]: !tt.ptr<i32>):
// CHECK:         [[NEW_PTR1:%.+]] = tt.addptr [[IN_PTR]], [[IN_I32]] : !tt.ptr<i32>, i32
// CHECK:         linalg.yield [[NEW_PTR1]] : !tt.ptr<i32>
// CHECK:     } -> tensor<128x!tt.ptr<i32>>
// CHECK:     tt.store [[ADDPTR128_1]], [[SLICE_LHS]] : tensor<128x!tt.ptr<i32>>
// CHECK:     [[EMPTY_PTR128_2:%.+]] = tensor.empty() : tensor<128x!tt.ptr<i32>>
// CHECK:     [[SPLAT_ARG2:%.+]] = linalg.fill ins(%arg2 : !tt.ptr<i32>) outs([[EMPTY_PTR128_2]] : tensor<128x!tt.ptr<i32>>) -> tensor<128x!tt.ptr<i32>>
// CHECK:     [[ADDPTR128_2:%.+]] = linalg.generic {{.*}} ins([[SPLAT_ARG2]], [[RANGE128]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>) outs([[SPLAT_ARG2]] : tensor<128x!tt.ptr<i32>>) {
// CHECK:       ^bb0([[IN_PTR:%.+]]: !tt.ptr<i32>, [[IN_I32:%.+]]: i32, [[OUT_PTR:%.+]]: !tt.ptr<i32>):
// CHECK:         [[NEW_PTR2:%.+]] = tt.addptr [[IN_PTR]], [[IN_I32]] : !tt.ptr<i32>, i32
// CHECK:         linalg.yield [[NEW_PTR2]] : !tt.ptr<i32>
// CHECK:     } -> tensor<128x!tt.ptr<i32>>
// CHECK:     tt.store [[ADDPTR128_2]], [[SLICE_RHS]] : tensor<128x!tt.ptr<i32>>
// CHECK:     return
// CHECK:   }
// CHECK: }
