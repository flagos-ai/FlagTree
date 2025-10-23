// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
  // CHECK-LABEL: func @addptr_splat_addptr
  tt.func public @addptr_splat_addptr(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c32_i32 = arith.constant 32 : i32
    %c64_i32 = arith.constant 64 : i32
    %c1_i64 = arith.constant 1 : i64
    %c4_i64 = arith.constant 4 : i64
    %c16_i64 = arith.constant 16 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.make_tensor_ptr %arg1, [%c4_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %1] {order = array<i32: 0, 1>} : !tt.ptr<tensor<4x8xf16>>
    %3 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %5 = tt.addptr %arg0, %1 : !tt.ptr<f16>, i32
    // CHECK: %[[OFFSET1:.*]] = arith.index_cast %[[BASE1:.*]] : i32 to index
    %6 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %2) -> !tt.ptr<tensor<4x8xf16>>  : i32 {
      %7 = tt.load %arg4 : !tt.ptr<tensor<4x8xf16>>
      %8 = tt.advance %arg4, [%c0_i32, %c32_i32] : <tensor<4x8xf16>>
      %9 = tt.reshape %7 : tensor<4x8xf16> -> tensor<32xf16>
      // CHECK: %[[IV_OFFSET:.*]] = arith.muli %[[IV:.*]], %c32_i32 : i32
      // CHECK: %[[OFFSET2:.*]] = arith.index_cast %[[IV_OFFSET]] : i32 to index
      // CHECK: %[[OFFSET:.*]] = arith.addi %[[OFFSET1]], %[[OFFSET2]] : index
      // CHECK: %[[CAST:.*]] = memref.reinterpret_cast %[[ARG0:.*]] to offset: [%[[OFFSET]]], sizes: [32], strides: [1] : memref<?xf16> to memref<32xf16, strided<[1], offset: ?>>
      // CHECK: bufferization.materialize_in_destination %[[RESHAPE:.*]] in writable %[[CAST]] : (tensor<32xf16>, memref<32xf16, strided<[1], offset: ?>>) -> ()
      %10 = arith.muli %arg3, %c32_i32 : i32
      %11 = tt.addptr %5, %10 : !tt.ptr<f16>, i32
      %12 = tt.splat %11 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
      %13 = tt.addptr %12, %3 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
      tt.store %13, %9 : tensor<32x!tt.ptr<f16>>
      scf.yield %8 : !tt.ptr<tensor<4x8xf16>>
    }
    tt.return
  }
}
