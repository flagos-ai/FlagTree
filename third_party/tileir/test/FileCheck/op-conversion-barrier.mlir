// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(convert-triton-to-cuda-tile,cuda_tile.module(cuda_tile.experimental\$func(fuse-fma)),reconcile-unrealized-casts,auto-gen-memory-token)" | FileCheck %s

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
// CHECK: %[[RESULT1:.*]], %[[TOKEN1:.*]] = load_ptr_tko
// CHECK: %[[RESULT2:.*]], %[[TOKEN2:.*]] = load_ptr_tko {{.*}}token=%[[TOKEN1]]
// CHECK: %[[TOKEN3:.*]] = store_ptr_tko {{.*}}token=%[[TOKEN2]]

// -----

module @test_barrier_layer_norm_bwd{
  tt.func public @test_barrier_layer_norm_bwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg7: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf16>
    %true = arith.constant true
    %c1_i32 = arith.constant 1 : i32
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1024xf32>
    %c0_i32 = arith.constant 0 : i32
    %c256_i32 = arith.constant 256 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %2 = tt.splat %arg10 : i32 -> tensor<1024xi32>
    %3 = arith.cmpi slt, %1, %2 : tensor<1024xi32>
    %4 = arith.muli %0, %arg9 : i32
    %5 = tt.addptr %arg4, %4 : !tt.ptr<f16>, i32
    %6 = tt.addptr %arg1, %4 : !tt.ptr<f16>, i32
    %7 = tt.addptr %arg0, %4 : !tt.ptr<f16>, i32
    %8 = arith.remsi %0, %c256_i32 : i32
    %9 = tt.addptr %arg8, %8 : !tt.ptr<i32>, i32
    %10 = tt.addptr %9, %c256_i32 : !tt.ptr<i32>, i32
    %11 = arith.muli %8, %arg10 : i32
    %12 = tt.addptr %arg2, %11 : !tt.ptr<f16>, i32
    %13 = tt.splat %12 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %14 = tt.addptr %13, %1 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %15 = tt.addptr %arg3, %11 : !tt.ptr<f16>, i32
    %16 = tt.splat %15 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %17 = tt.addptr %16, %1 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %18 = tt.splat %5 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %19 = tt.addptr %18, %1 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %20 = tt.load %19, %3, %cst : tensor<1024x!tt.ptr<f16>>
    %21 = arith.extf %20 : tensor<1024xf16> to tensor<1024xf32>
    %22 = tt.splat %6 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %23 = tt.addptr %22, %1 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %24 = tt.load %23, %3, %cst : tensor<1024x!tt.ptr<f16>>
    %25 = arith.extf %24 : tensor<1024xf16> to tensor<1024xf32>
    %26 = tt.splat %arg5 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %27 = tt.addptr %26, %1 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %28 = tt.load %27, %3 : tensor<1024x!tt.ptr<f16>>
    %29 = arith.extf %28 : tensor<1024xf16> to tensor<1024xf32>
    %30 = tt.addptr %arg6, %0 : !tt.ptr<f32>, i32
    %31 = tt.load %30 : !tt.ptr<f32>
    %32 = tt.addptr %arg7, %0 : !tt.ptr<f32>, i32
    %33 = tt.load %32 : !tt.ptr<f32>
    %34 = tt.splat %31 : f32 -> tensor<1024xf32>
    %35 = arith.subf %21, %34 : tensor<1024xf32>
    %36 = tt.splat %33 : f32 -> tensor<1024xf32>
    %37 = arith.mulf %35, %36 : tensor<1024xf32>
    %38 = arith.mulf %29, %25 : tensor<1024xf32>
    %39 = arith.select %3, %37, %cst_0 : tensor<1024xi1>, tensor<1024xf32>
    %40 = arith.select %3, %38, %cst_0 : tensor<1024xi1>, tensor<1024xf32>
    %41 = arith.mulf %39, %40 : tensor<1024xf32>
    %42 = "tt.reduce"(%41) <{axis = 0 : i32}> ({
    ^bb0(%arg11: f32, %arg12: f32):
      %63 = arith.addf %arg11, %arg12 : f32
      tt.reduce.return %63 : f32
    }) : (tensor<1024xf32>) -> f32
    %43 = arith.sitofp %arg10 : i32 to f32
    %44 = arith.divf %42, %43 : f32
    %45 = "tt.reduce"(%40) <{axis = 0 : i32}> ({
    ^bb0(%arg11: f32, %arg12: f32):
      %63 = arith.addf %arg11, %arg12 : f32
      tt.reduce.return %63 : f32
    }) : (tensor<1024xf32>) -> f32
    %46 = arith.divf %45, %43 : f32
    %47 = tt.splat %44 : f32 -> tensor<1024xf32>
    %48 = arith.mulf %39, %47 : tensor<1024xf32>
    %49 = tt.splat %46 : f32 -> tensor<1024xf32>
    %50 = arith.addf %48, %49 : tensor<1024xf32>
    %51 = arith.subf %40, %50 : tensor<1024xf32>
    %52 = arith.mulf %51, %36 : tensor<1024xf32>
    %53 = tt.splat %7 : !tt.ptr<f16> -> tensor<1024x!tt.ptr<f16>>
    %54 = tt.addptr %53, %1 : tensor<1024x!tt.ptr<f16>>, tensor<1024xi32>
    %55 = arith.truncf %52 : tensor<1024xf32> to tensor<1024xf16>
    tt.store %54, %55, %3 : tensor<1024x!tt.ptr<f16>>
    %56 = arith.mulf %25, %39 : tensor<1024xf32>
    scf.while : () -> () {
      %63 = tt.atomic_cas acq_rel, gpu, %9, %c0_i32, %c1_i32 : (!tt.ptr<i32>, i32, i32) -> i32
      %64 = arith.cmpi eq, %63, %c1_i32 : i32
      scf.condition(%64)
    } do {
      scf.yield
    }
    %57 = tt.load %10 : !tt.ptr<i32>
    %58 = arith.cmpi eq, %57, %c0_i32 : i32
    %59:2 = scf.if %58 -> (tensor<1024xf32>, tensor<1024xf32>) {
      %63 = tt.atomic_rmw exch, acq_rel, gpu, %10, %c1_i32, %true : (!tt.ptr<i32>, i32, i1) -> i32
      scf.yield %56, %25 : tensor<1024xf32>, tensor<1024xf32>
    } else {
      %63 = tt.load %14, %3 : tensor<1024x!tt.ptr<f16>>
      %64 = arith.extf %63 : tensor<1024xf16> to tensor<1024xf32>
      %65 = arith.addf %56, %64 : tensor<1024xf32>
      %66 = tt.load %17, %3 : tensor<1024x!tt.ptr<f16>>
      %67 = arith.extf %66 : tensor<1024xf16> to tensor<1024xf32>
      %68 = arith.addf %25, %67 : tensor<1024xf32>
      scf.yield %65, %68 : tensor<1024xf32>, tensor<1024xf32>
    }
    %60 = arith.truncf %59#0 : tensor<1024xf32> to tensor<1024xf16>
    tt.store %14, %60, %3 : tensor<1024x!tt.ptr<f16>>
    %61 = arith.truncf %59#1 : tensor<1024xf32> to tensor<1024xf16>
    tt.store %17, %61, %3 : tensor<1024x!tt.ptr<f16>>
    gpu.barrier
    %62 = tt.atomic_rmw exch, acq_rel, gpu, %9, %c0_i32, %true : (!tt.ptr<i32>, i32, i1) -> i32
    tt.return
  }
}

// CHECK-LABEL: @test_barrier_layer_norm_bwd
// CHECK-NOT: gpu.barrier
// CHECK: %[[RESULT1:.*]], %[[TOKEN1:.*]] = load_ptr_tko
// CHECK: %[[RESULT2:.*]], %[[TOKEN2:.*]] = load_ptr_tko {{.*}}token=%[[TOKEN1]]
// CHECK: %[[RESULT3:.*]], %[[TOKEN3:.*]] = load_ptr_tko {{.*}}token=%[[TOKEN2]]
// CHECK: %[[RESULT4:.*]], %[[TOKEN4:.*]] = load_ptr_tko {{.*}}token=%[[TOKEN3]]
// CHECK: %[[RESULT5:.*]], %[[TOKEN5:.*]] = load_ptr_tko {{.*}}token=%[[TOKEN4]]
// CHECK: %[[TOKEN6:.*]] = store_ptr_tko {{.*}}token=%[[TOKEN5]]
// CHECK: %[[TOKEN8:.*]] = loop iter_values(%[[TOKEN_LOOP:.*]] = %[[TOKEN6]]) : token -> token {
// CHECK:   %[[RESULT7:.*]], %[[TOKEN7:.*]] = atomic_cas_tko {{.*}} token=%[[TOKEN_LOOP]]
// CHECK:   break %[[TOKEN7]] : token
// CHECK: }
// CHECK: %[[RESULT9:.*]], %[[TOKEN9:.*]] = load_ptr_tko {{.*}}token=%[[TOKEN8]]
// CHECK: %[[IF:.*]]:3 = if %{{.*}} {
// CHECK:   %[[RMW_RES:.*]], %[[RMW_TOKEN:.*]] = atomic_rmw_tko {{.*}} token=%[[TOKEN9]]
// CHECK:   yield %{{.*}}, %{{.*}}, %[[RMW_TOKEN]]
// CHECK: } else {
// CHECK:   %[[LOAD1_RES:.*]], %[[LOAD1_TOKEN:.*]] = load_ptr_tko {{.*}} token=%[[TOKEN9]]
// CHECK:   %[[LOAD2_RES:.*]], %[[LOAD2_TOKEN:.*]] = load_ptr_tko {{.*}} token=%[[LOAD1_TOKEN]]
// CHECK:   yield %{{.*}}, %{{.*}}, %[[LOAD2_TOKEN]]
// CHECK: }
// CHECK: %[[TOKEN10:.*]] = store_ptr_tko {{.*}} token=%[[IF]]#2
// CHECK: %[[TOKEN11:.*]] = store_ptr_tko {{.*}} token=%[[TOKEN10]]
// CHECK: %[[RESULT12:.*]], %[[TOKEN12:.*]] = atomic_rmw_tko {{.*}} token=%[[TOKEN11]]
