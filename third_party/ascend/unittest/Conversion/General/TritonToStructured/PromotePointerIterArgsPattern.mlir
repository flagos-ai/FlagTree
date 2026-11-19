// RUN: triton-opt  '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true'  --split-input-file %s | FileCheck %s

module {
  tt.func public @test_promote_pointer_iter(%base_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> !tt.ptr<f32> {
    %c1_i32 = arith.constant 1 : i32
    %c0_f32 = arith.constant 0.000000e+00 : f32
    %true_mask = arith.constant 1 : i1
    %c0_index = arith.constant 0 : index
    %c10_index = arith.constant 10 : index
    %c1_index = arith.constant 1 : index
    %final_ptr = scf.for %iv = %c0_index to %c10_index step %c1_index iter_args(%ptr = %base_ptr) -> (!tt.ptr<f32>) {
      %data = tt.load %ptr, %true_mask, %c0_f32 : !tt.ptr<f32>
      %new_ptr = tt.addptr %ptr, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %new_ptr : !tt.ptr<f32>
    }
    tt.return %final_ptr : !tt.ptr<f32>
  }
}

// CHECK-LABEL:   tt.func public @test_promote_pointer_iter(
// CHECK-SAME:                                %[[VAL_0:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) -> !tt.ptr<f32> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = scf.for %[[VAL_6:.*]] = %[[VAL_2]] to %[[VAL_3]] step %[[VAL_4]] iter_args(%[[VAL_7:.*]] = %[[VAL_0]]) -> (!tt.ptr<f32>) {
// CHECK:             %[[VAL_8:.*]] = tt.addptr %[[VAL_7]], %[[VAL_1]] : !tt.ptr<f32>, i32
// CHECK:             scf.yield %[[VAL_8]] : !tt.ptr<f32>
// CHECK:           }
// CHECK:           tt.return %[[VAL_5]] : !tt.ptr<f32>
// CHECK:         }


// -----


module {
  tt.func public @test_promote_pointer_iter_advance(%base_ptr: !tt.ptr<f16>) -> !tt.ptr<tensor<32xf16>>{
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %c1_i32 = arith.constant 1 : i32  // nonZeroConstant 需要 1
    %c0_i32_2 = arith.constant 0 : i32
    %c0_index = arith.constant 0 : index
    %c10_index = arith.constant 10 : index
    %c1_index = arith.constant 1 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<32xf16>
    %ptr0 = tt.make_tensor_ptr %base_ptr, [%c32_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>} : !tt.ptr<tensor<32xf16>>
    %final_ptr = scf.for %iv = %c0_index to %c10_index step %c1_index iter_args(%ptr = %ptr0) -> !tt.ptr<tensor<32xf16>> {
      %data = tt.load %ptr : !tt.ptr<tensor<32xf16>>
      %new_ptr = tt.advance %ptr, [%c1_i32, %c0_i32_2] : !tt.ptr<tensor<32xf16>>
      scf.yield %new_ptr : !tt.ptr<tensor<32xf16>>
    }
    tt.return %final_ptr : !tt.ptr<tensor<32xf16>>
  }
}

// CHECK-LABEL:   tt.func public @test_promote_pointer_iter_advance(
// CHECK-SAME:                                %[[VAL_0:.*]]: !tt.ptr<f16>) -> !tt.ptr<tensor<32xf16>> {
// CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_3:.*]] = arith.constant 32 : i64
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 10 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_8:.*]] = tt.make_tensor_ptr %[[VAL_0]], [%[[VAL_3]]], [%[[VAL_2]]], [%[[VAL_1]]] {order = array<i32: 0>} : <tensor<32xf16>>
// CHECK:           %[[VAL_9:.*]] = scf.for %[[VAL_10:.*]] = %[[VAL_5]] to %[[VAL_6]] step %[[VAL_7]] iter_args(%[[VAL_11:.*]] = %[[VAL_8]]) -> (!tt.ptr<tensor<32xf16>>) {
// CHECK:             %[[VAL_12:.*]] = tt.advance %[[VAL_11]], [%[[VAL_4]], %[[VAL_1]]] : <tensor<32xf16>>
// CHECK:             scf.yield %[[VAL_12]] : !tt.ptr<tensor<32xf16>>
// CHECK:           }
// CHECK:           tt.return %[[VAL_9]] : !tt.ptr<tensor<32xf16>>
// CHECK:         }
