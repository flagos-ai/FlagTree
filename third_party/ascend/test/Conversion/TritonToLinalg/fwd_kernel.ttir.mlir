// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @_fwd_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c256_i32 = arith.constant 256 : i32
    %cst = arith.constant dense<16> : tensor<16x1xi32>
    %cst_0 = arith.constant dense<0xFF800000> : tensor<16x16xf32>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %cst_2 = arith.constant dense<0> : tensor<16x16xi32>
    %cst_3 = arith.constant dense<16> : tensor<1x16xi32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %c16_i32 = arith.constant 16 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.remsi %0, %c16_i32 : i32
    %2 = tt.get_program_id y : i32
    %3 = arith.muli %0, %c256_i32 : i32
    %4 = arith.muli %2, %c16_i32 : i32
    %5 = tt.addptr %arg0, %3 : !tt.ptr<f32>, i32
    %6 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %7 = tt.expand_dims %6 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %8 = tt.splat %5 : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %9 = tt.addptr %8, %7 : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %10 = tt.addptr %arg1, %3 : !tt.ptr<f32>, i32
    %11 = tt.expand_dims %6 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %12 = tt.splat %10 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %13 = tt.addptr %12, %11 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %14 = tt.addptr %arg2, %3 : !tt.ptr<f32>, i32
    %15 = tt.addptr %14, %4 : !tt.ptr<f32>, i32
    %16 = tt.splat %15 : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %17 = tt.addptr %16, %7 : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %18 = tt.addptr %arg3, %3 : !tt.ptr<f32>, i32
    %19 = tt.addptr %18, %4 : !tt.ptr<f32>, i32
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %21 = tt.addptr %20, %7 : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %22 = tt.addptr %arg4, %1 : !tt.ptr<f32>, i32
    %23 = tt.load %22 : !tt.ptr<f32>
    %24 = arith.subf %cst_4, %23 : f32
    %25 = arith.sitofp %11 : tensor<16x1xi32> to tensor<16x1xf32>
    %26 = tt.splat %24 : f32 -> tensor<16x1xf32>
    %27 = arith.mulf %26, %25 : tensor<16x1xf32>
    %28 = math.exp %27 : tensor<16x1xf32>
    %29 = tt.broadcast %11 : tensor<16x1xi32> -> tensor<16x16xi32>
    %30 = tt.broadcast %7 : tensor<1x16xi32> -> tensor<16x16xi32>
    %31 = arith.subi %29, %30 : tensor<16x16xi32>
    %32 = arith.sitofp %31 : tensor<16x16xi32> to tensor<16x16xf32>
    %33 = tt.splat %23 : f32 -> tensor<16x16xf32>
    %34 = arith.mulf %33, %32 : tensor<16x16xf32>
    %35 = arith.cmpi sge, %31, %cst_2 : tensor<16x16xi32>
    %36 = arith.subf %cst_1, %34 : tensor<16x16xf32>
    %37 = arith.select %35, %36, %cst_0 : tensor<16x16xi1>, tensor<16x16xf32>
    %38 = math.exp %37 : tensor<16x16xf32>
    %39 = arith.cmpi slt, %11, %cst : tensor<16x1xi32>
    %40 = arith.muli %11, %cst : tensor<16x1xi32>
    %41 = tt.broadcast %9 : tensor<1x16x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %42 = tt.broadcast %40 : tensor<16x1xi32> -> tensor<16x16xi32>
    %43 = tt.addptr %41, %42 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %44 = tt.broadcast %39 : tensor<16x1xi1> -> tensor<16x16xi1>
    %45 = tt.load %43, %44, %cst_1 : tensor<16x16x!tt.ptr<f32>>
    %46 = arith.cmpi slt, %7, %cst_3 : tensor<1x16xi32>
    %47 = arith.muli %7, %cst_3 : tensor<1x16xi32>
    %48 = tt.broadcast %13 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %49 = tt.broadcast %47 : tensor<1x16xi32> -> tensor<16x16xi32>
    %50 = tt.addptr %48, %49 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %51 = tt.broadcast %46 : tensor<1x16xi1> -> tensor<16x16xi1>
    %52 = tt.load %50, %51, %cst_1 : tensor<16x16x!tt.ptr<f32>>
    %53 = tt.broadcast %17 : tensor<1x16x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %54 = tt.addptr %53, %42 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %55 = tt.load %54, %44, %cst_1 : tensor<16x16x!tt.ptr<f32>>
    %56 = tt.dot %45, %52, %cst_1 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %57 = arith.mulf %56, %38 : tensor<16x16xf32>
    %58 = tt.dot %45, %cst_1, %cst_1 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %59 = tt.broadcast %28 : tensor<16x1xf32> -> tensor<16x16xf32>
    %60 = arith.mulf %58, %59 : tensor<16x16xf32>
    %61 = tt.dot %57, %55, %60 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    %62 = tt.broadcast %21 : tensor<1x16x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %63 = tt.addptr %62, %42 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %63, %61, %44 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}

//CHECK: func.func @fwd_kernel
//CHECK: %[[ALLOC:alloc[0-9]+]] = memref.alloc() : memref<16x16xf32>
//CHECK: annotation.mark %[[ALLOC]] {MayImplicitTransposeWithLastAxis} : memref<16x16xf32>
