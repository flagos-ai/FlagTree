// RUN: triton-adapter-opt --triton-to-annotation --triton-to-linalg %s | FileCheck %s
module {
  // CHECK-LABEL: func @permute_3d
  tt.func public @permute_3d(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c12_i64 = arith.constant 12 : i64
    %c512_i64 = arith.constant 512 : i64
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32: i32
    %2 = tt.make_tensor_ptr %arg1, [%c12_i64, %c512_i64], [%c512_i64, %c1_i64], [%c0_i32, %1] {order = array<i32: 0, 1>} : !tt.ptr<tensor<12x512xf16>>
    %3 = tt.load %2 : !tt.ptr<tensor<12x512xf16>>
    // CHECK-NOT: annotation.mark %[[LOADED:.*]] {MayImplicitTransposeWithLastAxis} : tensor<512x12xf16>
    %4 = tt.reshape %3 : tensor<12x512xf16> -> tensor<12x4x128xf16>
    // CHECK: %[[RES:.*]] = tensor.empty() : tensor<4x12x128xf16>
    // CHECK: %[[TRANS:.*]] = linalg.transpose ins(%[[SRC:.*]] : tensor<12x4x128xf16>) outs(%[[RES]] : tensor<4x12x128xf16>) permutation = [1, 0, 2]
    %5 = tt.trans %4 {order = array<i32: 1, 0, 2>} : tensor<12x4x128xf16> -> tensor<4x12x128xf16>
    %6 = tt.reshape %5 : tensor<4x12x128xf16> -> tensor<6144xf16>
    %7 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<6144x!tt.ptr<f16>>
    %8 = tt.make_range {end = 6144 : i32, start = 0 : i32} : tensor<6144xi32>
    %9 = tt.addptr %7, %8 : tensor<6144x!tt.ptr<f16>>, tensor<6144xi32>
    tt.store %9, %6 evictionPolicy = evict_last : tensor<6144x!tt.ptr<f16>>
    tt.return
  }
}
