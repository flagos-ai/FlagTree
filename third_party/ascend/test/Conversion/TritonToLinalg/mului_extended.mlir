// RUN: triton-adapter-opt %s --triton-to-linalg  | FileCheck %s

module {
  tt.func public @umulhi_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32},
  %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %3 = tt.load %2 : tensor<128x!tt.ptr<i32>>
    %4 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %5 = tt.addptr %4, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %6 = tt.load %5 : tensor<128x!tt.ptr<i32>>
    %7 = tt.mulhiui %3, %6 : tensor<128xi32>
    %8 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    tt.store %9, %7 : tensor<128x!tt.ptr<i32>>
    tt.return
  }
}

//CHECK-LABEL: @umulhi_kernel
//CHECK: %[[VAL0:.*]] = bufferization.to_tensor %alloc restrict writable : memref<128xi32>
//CHECK: %[[VAL1:.*]] = bufferization.to_tensor %alloc_1 restrict writable : memref<128xi32>
//CHECK: %[[VAL2:.*]], %[[VAL3:.*]] = arith.mulsi_extended %in, %in_3 : i32

