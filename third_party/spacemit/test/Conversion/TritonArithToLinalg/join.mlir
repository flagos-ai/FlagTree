// RUN: triton-shared-opt --triton-arith-to-linalg %s | FileCheck %s

module {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
    %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<128x1xi32>
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %3 = tt.load %2 : tensor<128x!tt.ptr<i32>>
    %4 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %6 = tt.load %5 : tensor<128x!tt.ptr<i32>>
    %7 = tt.join %3, %6 : tensor<128xi32> -> tensor<128x2xi32>
    %8 = tt.expand_dims %0 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %9 = arith.muli %8, %cst : tensor<128x1xi32>
    %10 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x1x!tt.ptr<i32>>
    %11 = tt.addptr %10, %9 : tensor<128x1x!tt.ptr<i32>>, tensor<128x1xi32>
    %12 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %14 = tt.broadcast %11 : tensor<128x1x!tt.ptr<i32>> -> tensor<128x2x!tt.ptr<i32>>
    %15 = tt.broadcast %13 : tensor<1x2xi32> -> tensor<128x2xi32>
    %16 = tt.addptr %14, %15 : tensor<128x2x!tt.ptr<i32>>, tensor<128x2xi32>
    tt.store %16, %7 : tensor<128x2x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK:   func.func @kernel(%arg0: !tt.ptr<i32> {{.*}}, %arg1: !tt.ptr<i32> {{.*}}, %arg2: !tt.ptr<i32> {{.*}}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
// CHECK:     [[C2_I32:%.+]] = arith.constant 2 : i32
// CHECK:     [[EMPTY_128X1_I32:%.+]] = tensor.empty() : tensor<128x1xi32>
// CHECK:     [[FILLED_C2:%.+]] = linalg.fill ins([[C2_I32]] : i32) outs([[EMPTY_128X1_I32]] : tensor<128x1xi32>) -> tensor<128x1xi32>
// CHECK:     [[EMPTY_128_I32:%.+]] = tensor.empty() : tensor<128xi32>
// CHECK:     [[RANGE_128:%.+]] = linalg.generic {{.*}} outs([[EMPTY_128_I32]] : tensor<128xi32>) {
// CHECK:       ^bb0(%out: i32):
// CHECK:         [[IDX0:%.+]] = linalg.index 0 : index
// CHECK:         [[I32_IDX0:%.+]] = arith.index_cast [[IDX0]] : index to i32
// CHECK:         linalg.yield [[I32_IDX0]] : i32
// CHECK:     } -> tensor<128xi32>
// CHECK:     [[EMPTY_PTR128:%.+]] = tensor.empty() : tensor<128x!tt.ptr<i32>>
// CHECK:     [[SPLAT_ARG0:%.+]] = linalg.fill ins(%arg0 : !tt.ptr<i32>) outs([[EMPTY_PTR128]] : tensor<128x!tt.ptr<i32>>) -> tensor<128x!tt.ptr<i32>>
// CHECK:     [[ADDPTR_ARG0:%.+]] = linalg.generic {{.*}} ins([[SPLAT_ARG0]], [[RANGE_128]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>) outs([[SPLAT_ARG0]] : tensor<128x!tt.ptr<i32>>) {
// CHECK:       ^bb0([[IN_PTR:%.+]]: !tt.ptr<i32>, [[IN_I32:%.+]]: i32, [[OUT_PTR:%.+]]: !tt.ptr<i32>):
// CHECK:         [[NEW_PTR0:%.+]] = tt.addptr [[IN_PTR]], [[IN_I32]] : !tt.ptr<i32>, i32
// CHECK:         linalg.yield [[NEW_PTR0]] : !tt.ptr<i32>
// CHECK:     } -> tensor<128x!tt.ptr<i32>>
// CHECK:     [[LOADED_ARG0:%.+]] = tt.load [[ADDPTR_ARG0]] : tensor<128x!tt.ptr<i32>>
// CHECK:     [[EMPTY_PTR128_1:%.+]] = tensor.empty() : tensor<128x!tt.ptr<i32>>
// CHECK:     [[SPLAT_ARG1:%.+]] = linalg.fill ins(%arg1 : !tt.ptr<i32>) outs([[EMPTY_PTR128_1]] : tensor<128x!tt.ptr<i32>>) -> tensor<128x!tt.ptr<i32>>
// CHECK:     [[ADDPTR_ARG1:%.+]] = linalg.generic {{.*}} ins([[SPLAT_ARG1]], [[RANGE_128]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>) outs([[SPLAT_ARG1]] : tensor<128x!tt.ptr<i32>>) {
// CHECK:       ^bb0([[IN_PTR:%.+]]: !tt.ptr<i32>, [[IN_I32:%.+]]: i32, [[OUT_PTR:%.+]]: !tt.ptr<i32>):
// CHECK:         [[NEW_PTR1:%.+]] = tt.addptr [[IN_PTR]], [[IN_I32]] : !tt.ptr<i32>, i32
// CHECK:         linalg.yield [[NEW_PTR1]] : !tt.ptr<i32>
// CHECK:     } -> tensor<128x!tt.ptr<i32>>
// CHECK:     [[LOADED_ARG1:%.+]] = tt.load [[ADDPTR_ARG1]] : tensor<128x!tt.ptr<i32>>
// CHECK:     [[EMPTY_JOIN:%.+]] = tensor.empty() : tensor<128x2xi32>
// CHECK:     [[INSERTED_SLICE0:%.+]] = tensor.insert_slice [[LOADED_ARG0]] into [[EMPTY_JOIN]]{{\[}}0, 0{{\]}} [128, 1] [1, 1] : tensor<128xi32> into tensor<128x2xi32>
// CHECK:     [[INSERTED_SLICE1:%.+]] = tensor.insert_slice [[LOADED_ARG1]] into [[INSERTED_SLICE0]]{{\[}}0, 1{{\]}} [128, 1] [1, 1] : tensor<128xi32> into tensor<128x2xi32>
// CHECK:     [[EXPANDED_RANGE:%.+]] = tensor.expand_shape
// CHECK:     [[MULI_RESULT:%.+]] = linalg.generic {{.*}} ins([[EXPANDED_RANGE]], [[FILLED_C2]] : tensor<128x1xi32>, tensor<128x1xi32>) outs([[EXPANDED_RANGE:%.+]] : tensor<128x1xi32>) {
// CHECK:       ^bb0([[IN_I32_0:%.+]]: i32, [[IN_I32_1:%.+]]: i32, %out: i32):
// CHECK:         [[MUL_RESULT:%.+]] = arith.muli [[IN_I32_0]], [[IN_I32_1]] : i32
// CHECK:         linalg.yield [[MUL_RESULT]] : i32
// CHECK:     } -> tensor<128x1xi32>
// CHECK:     [[EMPTY_PTR128X1:%.+]] = tensor.empty() : tensor<128x1x!tt.ptr<i32>>
// CHECK:     [[SPLAT_ARG2:%.+]] = linalg.fill ins(%arg2 : !tt.ptr<i32>) outs([[EMPTY_PTR128X1]] : tensor<128x1x!tt.ptr<i32>>) -> tensor<128x1x!tt.ptr<i32>>
// CHECK:     [[ADDPTR_ARG2:%.+]] = linalg.generic {{.*}} ins([[SPLAT_ARG2]], [[MULI_RESULT]] : tensor<128x1x!tt.ptr<i32>>, tensor<128x1xi32>) outs([[SPLAT_ARG2]] : tensor<128x1x!tt.ptr<i32>>) {
// CHECK:       ^bb0([[IN_PTR:%.+]]: !tt.ptr<i32>, [[IN_I32:%.+]]: i32, [[OUT_PTR:%.+]]: !tt.ptr<i32>):
// CHECK:         [[NEW_PTR2:%.+]] = tt.addptr [[IN_PTR]], [[IN_I32]] : !tt.ptr<i32>, i32
// CHECK:         linalg.yield [[NEW_PTR2]] : !tt.ptr<i32>
// CHECK:     } -> tensor<128x1x!tt.ptr<i32>>
// CHECK:     [[EMPTY_RANGE2:%.+]] = tensor.empty() : tensor<2xi32>
// CHECK:     [[RANGE_2:%.+]] = linalg.generic {{.*}} outs([[EMPTY_RANGE2]] : tensor<2xi32>) {
// CHECK:       ^bb0(%out: i32):
// CHECK:         [[IDX1:%.+]] = linalg.index 0 : index
// CHECK:         [[I32_IDX1:%.+]] = arith.index_cast [[IDX1]] : index to i32
// CHECK:         linalg.yield [[I32_IDX1]] : i32
// CHECK:     } -> tensor<2xi32>
// CHECK:     [[EXPANDED_RANGE2:%.+]] = tensor.expand_shape [[RANGE_2]]
// CHECK:     [[EMPTY_PTR128X2:%.+]] = tensor.empty() : tensor<128x2x!tt.ptr<i32>>
// CHECK:     [[BROADCASTED_PTR:%.+]] = linalg.generic {{.*}} ins([[ADDPTR_ARG2]] : tensor<128x1x!tt.ptr<i32>>) outs([[EMPTY_PTR128X2]] : tensor<128x2x!tt.ptr<i32>>) attrs = {{.*}} {
// CHECK:       ^bb0([[IN_PTR:%.+]]: !tt.ptr<i32>, [[OUT_PTR:%.+]]: !tt.ptr<i32>):
// CHECK:         linalg.yield [[IN_PTR]] : !tt.ptr<i32>
// CHECK:     } -> tensor<128x2x!tt.ptr<i32>>
// CHECK:     [[EMPTY_I32_128X2:%.+]] = tensor.empty() : tensor<128x2xi32>
// CHECK:     [[BROADCASTED_I32:%.+]] = linalg.generic {{.*}} ins([[EXPANDED_RANGE2]] : tensor<1x2xi32>) outs([[EMPTY_I32_128X2]] : tensor<128x2xi32>) attrs = {{.*}} {
// CHECK:       ^bb0([[IN_I32:%.+]]: i32, [[OUT_I32:%.+]]: i32):
// CHECK:         linalg.yield [[IN_I32]] : i32
// CHECK:     } -> tensor<128x2xi32>
// CHECK:     [[ADDPTR_FINAL:%.+]] = linalg.generic {{.*}} ins([[BROADCASTED_PTR]], [[BROADCASTED_I32]] : tensor<128x2x!tt.ptr<i32>>, tensor<128x2xi32>) outs([[BROADCASTED_PTR]] : tensor<128x2x!tt.ptr<i32>>) {
// CHECK:       ^bb0([[IN_PTR:%.+]]: !tt.ptr<i32>, [[IN_I32:%.+]]: i32, [[OUT_PTR:%.+]]: !tt.ptr<i32>):
// CHECK:         [[FINAL_PTR:%.+]] = tt.addptr [[IN_PTR]], [[IN_I32]] : !tt.ptr<i32>, i32
// CHECK:         linalg.yield [[FINAL_PTR]] : !tt.ptr<i32>
// CHECK:     } -> tensor<128x2x!tt.ptr<i32>>
// CHECK:     tt.store [[ADDPTR_FINAL]], [[INSERTED_SLICE1]] : tensor<128x2x!tt.ptr<i32>>
