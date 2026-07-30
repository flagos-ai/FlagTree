// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @gather_simple(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<64> : tensor<64xi32>
    %c64_i32 = arith.constant 64 : i32
    %c5_i32 = arith.constant 5 : i32
    %cst_0 = arith.constant dense<10> : tensor<64xi32>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<64xi32>, tensor<64xi32>)  : i32 {
      %4 = arith.divsi %arg3, %cst_0 : tensor<64xi32>
      %5 = arith.addi %arg2, %c5_i32 : i32
      %6 = arith.remsi %5, %c64_i32 : i32
      %7 = tt.splat %6 : i32 -> tensor<64xi32>
      %8 = arith.addi %4, %7 : tensor<64xi32>
      %9 = tt.addptr %1, %8 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %10 = tt.load %9 : tensor<64x!tt.ptr<f32>>
      %11 = tt.addptr %2, %arg4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %11, %10 : tensor<64x!tt.ptr<f32>>
      %12 = arith.addi %8, %cst : tensor<64xi32>
      %13 = arith.addi %arg4, %cst : tensor<64xi32>
      scf.yield %12, %13 : tensor<64xi32>, tensor<64xi32>
    }
    tt.return
  }
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store

// CHECK: [[tensor:%.+]] = tts.gather %arg0
// CHECK: tts.scatter [[tensor]] into %arg1
