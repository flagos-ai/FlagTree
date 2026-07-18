// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(rewrite-assume-with-cuda-tile)" | FileCheck %s

// CHECK-LABEL: kernel_assume
// CHECK: [[V0:%.*]] = builtin.unrealized_conversion_cast {{.*}} : i32 to !cuda_tile.tile<i32>
// CHECK-NEXT: [[V1:%.*]] = cuda_tile.assume div_by<32>, [[V0]] : tile<i32>
// CHECK-NEXT: builtin.unrealized_conversion_cast [[V1]] : !cuda_tile.tile<i32> to i32 {tt.divisibility = 32 : i32}
module @kernel_assume {
  tt.func private @assume(%arg0: i32 {tt.divisibility = 16 : i32}) -> i32 {
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.remsi %arg0, %c32_i32 : i32
    %1 = arith.cmpi eq, %0, %c0_i32 : i32
    llvm.intr.assume %1 : i1
    %2 = arith.addi %arg0, %arg0 : i32
    tt.return %2 : i32
  }
}

// -----

// CHECK-LABEL: kernel_assume_2
// CHECK: tt.ptr_to_int [[V0:%.*]] {tt.divisibility = 32 : i64} : !tt.ptr<i32> -> i64
// CHECK-NEXT: [[V1:%.*]] = builtin.unrealized_conversion_cast [[V0]] : !tt.ptr<i32> to !cuda_tile.tile<ptr<i32>>
// CHECK-NEXT: cuda_tile.assume div_by<32>, [[V1]] : tile<ptr<i32>>
module @kernel_assume_2 {
  tt.func private @assume_2(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) -> i64 {
    %0 = tt.ptr_to_int %arg0 : !tt.ptr<i32> -> i64
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64
    %1 = arith.remsi %0, %c32_i64 : i64
    %2 = arith.cmpi eq, %1, %c0_i64 : i64
    llvm.intr.assume %2 : i1
    %3 = arith.addi %0, %0 : i64
    tt.return %3 : i64
  }
}

// -----

// CHECK-LABEL: kernel_assume_3
// CHECK: [[V0:%.*]] = arith.addi {{.*}}, {{.*}} {tt.divisibility = 32 : i64} : i64
// CHECK-NEXT: [[V1:%.*]] = builtin.unrealized_conversion_cast [[V0]] : i64 to !cuda_tile.tile<i64>
// CHECK-NEXT: [[V2:%.*]] = cuda_tile.assume div_by<32>, [[V1]] : tile<i64>
// CHECK-NEXT: builtin.unrealized_conversion_cast [[V2]] : !cuda_tile.tile<i64> to i64 {tt.divisibility = 32 : i64}
module @kernel_assume_3 {
  tt.func private @assume_3(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) -> i64 {
    %0 = tt.ptr_to_int %arg0 : !tt.ptr<i32> -> i64
    %c32_i64 = arith.constant 32 : i64
    %1 = arith.addi %0, %c32_i64 : i64
    %c0_i64 = arith.constant 0 : i64
    %2 = arith.remsi %1, %c32_i64 : i64
    %3 = arith.cmpi eq, %2, %c0_i64 : i64
    llvm.intr.assume %3 : i1
    %4 = arith.addi %1, %1 : i64
    tt.return %4 : i64
  }
}
