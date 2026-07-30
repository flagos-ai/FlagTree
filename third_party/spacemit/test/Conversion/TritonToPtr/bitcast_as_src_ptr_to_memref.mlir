// RUN: triton-shared-opt --triton-arith-to-linalg="tensor-ptr-to-linalg" --triton-to-ptr --cse --canonicalize %s | FileCheck %s

module {
  tt.func public @bitcast_ptr_as_src(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<16xi32>
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.addptr %arg0, %c1_i32 : !tt.ptr<i32>, i32
    %1 = tt.bitcast %0 : !tt.ptr<i32> -> !tt.ptr<i64>
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.splat %1 : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>>
    %4 = tt.addptr %3, %2 : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    %5 = tt.addptr %4, %cst : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    %6 = tt.load %5 : tensor<16x!tt.ptr<i64>>
    %7 = tt.addptr %arg1, %c1_i32 : !tt.ptr<i32>, i32
    %8 = tt.bitcast %7 : !tt.ptr<i32> -> !tt.ptr<i64>
    %9 = tt.splat %8 : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>>
    %10 = tt.addptr %9, %2 : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    %11 = tt.addptr %10, %cst : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
    tt.store %11, %6 : tensor<16x!tt.ptr<i64>>
    tt.return
  }
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.store
// CHECK-NOT: tt.load
// CHECK-NOT: tt.bitcast
// CHECK-NOT: tt.make_range
// CHECK-NOT: tt.splat
