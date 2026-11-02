// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(convert-triton-to-cuda-tile,cuda_tile.module(cuda_tile.entry(fuse-fma)),reconcile-unrealized-casts,cuda_tile.module(cuda_tile.entry(auto-gen-memory-token)))" | FileCheck %s

// Keep this file focused on conversion-pipeline smoke coverage for
// gpu.barrier. Detailed barrier ordering is covered by
// auto-memtoken-fences-atomics.mlir.

module @test_barrier_add_kernel {
  tt.func public @test_barrier_add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %1 : i32 -> tensor<1024xi32>
    %4 = arith.addi %3, %2 : tensor<1024xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>>
    %13 = arith.addf %9, %12 : tensor<1024xf32>
    gpu.barrier
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: @test_barrier_add_kernel
// CHECK-NOT: gpu.barrier
// Two loads on distinct arg ptrs (%arg0, %arg1) are in distinct alias
// classes and have no inter-load token edge.
// CHECK: %{{.*}}, %[[TOKEN1:.*]] = load_ptr_tko
// CHECK-NOT: token=
// CHECK: %{{.*}}, %[[TOKEN2:.*]] = load_ptr_tko
// gpu.barrier is acq+rel: it joins every prior lastOp and installs
// the join as the input token for every following op (including the
// store on %arg2, a third distinct class).
// CHECK: %[[JOIN:.*]] = join_tokens %[[TOKEN1]], %[[TOKEN2]]
// CHECK: store_ptr_tko {{.*}}token=%[[JOIN]]
