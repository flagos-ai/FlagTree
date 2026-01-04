// Copyright (c) 2025  XCoreSigma Inc. All rights reserved.
// flagtree tle

// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:80 num-warps=4' | FileCheck %s

// Test TLE local_alloc and local_load operations conversion from Triton to TritonGPU
// CHECK: #blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK: #blocked1 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
// CHECK: #blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #blocked3 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
// CHECK: #blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
// CHECK: module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:80", "ttg.threads-per-warp" = 32 : i32}
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module {
  tt.func public @elementwise_add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %3 = tt.splat %1 : i32 -> tensor<128xi32>
    %4 = arith.addi %3, %2 : tensor<128xi32>
    %5 = tt.expand_dims %4 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %6 = tt.splat %arg5 : i32 -> tensor<128x1xi32>
    %7 = arith.muli %6, %5 : tensor<128x1xi32>
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
    %10 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %11 = arith.muli %10, %5 : tensor<128x1xi32>
    %12 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
    %13 = tt.addptr %12, %11 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
    %14 = tt.splat %arg7 : i32 -> tensor<128x1xi32>
    %15 = arith.muli %14, %5 : tensor<128x1xi32>
    %16 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<128x1x!tt.ptr<f32>>
    %17 = tt.addptr %16, %15 : tensor<128x1x!tt.ptr<f32>>, tensor<128x1xi32>
    scf.for %arg8 = %c0_i32 to %arg4 step %c128_i32  : i32 {
      %18 = tt.splat %arg8 : i32 -> tensor<128xi32>
      %19 = arith.addi %2, %18 : tensor<128xi32>
      %20 = tt.splat %arg3 : i32 -> tensor<128xi32>
      %21 = arith.cmpi slt, %4, %20 : tensor<128xi32>
      %22 = tt.expand_dims %21 {axis = 1 : i32} : tensor<128xi1> -> tensor<128x1xi1>
      %23 = tt.splat %arg4 : i32 -> tensor<128xi32>
      %24 = arith.cmpi slt, %19, %23 : tensor<128xi32>
      %25 = tt.expand_dims %24 {axis = 0 : i32} : tensor<128xi1> -> tensor<1x128xi1>
      %26 = tt.broadcast %22 : tensor<128x1xi1> -> tensor<128x128xi1>
      %27 = tt.broadcast %25 : tensor<1x128xi1> -> tensor<128x128xi1>
      %28 = arith.andi %26, %27 : tensor<128x128xi1>
      %29 = tt.expand_dims %19 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
      %30 = tt.broadcast %9 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
      %31 = tt.broadcast %29 : tensor<1x128xi32> -> tensor<128x128xi32>
      %32 = tt.addptr %30, %31 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      %33 = tt.load %32 : tensor<128x128x!tt.ptr<f32>>

    // Test TLE local_alloc operation conversion
    // CHECK: ttg.local_alloc
    // CHECK-SAME: !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %34 = ttg.local_alloc %33 : (tensor<128x128xf32>) -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>

    // Test TLE local_load operation conversion
    // CHECK: ttg.local_load
    // CHECK-SAME: !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %35 = ttg.local_load %34 : !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> tensor<128x128xf32>

    %36 = tt.broadcast %13 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
    %37 = tt.addptr %36, %31 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    %38 = tt.load %37 : tensor<128x128x!tt.ptr<f32>>

    // Test second TLE local_alloc and local_load
    // CHECK: ttg.local_alloc
    // CHECK-SAME: !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %39 = ttg.local_alloc %38 : (tensor<128x128xf32>) -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>

    // CHECK: ttg.local_load
    // CHECK-SAME: !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %40 = ttg.local_load %39 : !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> tensor<128x128xf32>
      %41 = arith.addf %35, %40 : tensor<128x128xf32>
      %42 = tt.broadcast %17 : tensor<128x1x!tt.ptr<f32>> -> tensor<128x128x!tt.ptr<f32>>
      %43 = tt.addptr %42, %31 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
      tt.store %43, %41, %28 : tensor<128x128x!tt.ptr<f32>>
    } {tt.num_stages = 2 : i32}
    tt.return
  }
}
