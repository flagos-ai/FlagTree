// RUN: triton-adapter-opt --triton-to-annotation --triton-to-unstructure  --bubble-up-operation --discrete-mask-access-conversion --triton-to-hivm "--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False" %s | FileCheck %s

module {
  // CHECK-LABEL:   func.func @_attn_fwd
  tt.func public @_attn_fwd(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg3: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg4: !tt.ptr<f16> {tt.divisibility = 16 : i32} , %arg5: f32 ) attributes {noinline = false} {
    %cst = arith.constant dense<1.000000e+00> : tensor<64xf32> 
    %cst_0 = arith.constant dense<0xFF800000> : tensor<64xf32> 
    %cst_1 = arith.constant dense<-1.000000e+06> : tensor<64x64xf32> 
    %cst_2 = arith.constant dense<0.72134751> : tensor<64xf32> 
    %cst_3 = arith.constant dense<0.000000e+00> : tensor<64x64xf32> 
    %c1_i32 = arith.constant 1 : i32 
    %c128_i32 = arith.constant 128 : i32 
    %c2048_i32 = arith.constant 2048 : i32 
    %cst_4 = arith.constant 1.44269502 : f32 
    %c1_i64 = arith.constant 1 : i64 
    %c64_i64 = arith.constant 64 : i64 
    %c2048_i64 = arith.constant 2048 : i64 
    %c64_i32 = arith.constant 64 : i32 
    %c131072_i64 = arith.constant 131072 : i64 
    %c4194304_i64 = arith.constant 4194304 : i64 
    %c32_i32 = arith.constant 32 : i32 
    %c0_i32 = arith.constant 0 : i32 
    %0 = tt.get_program_id x : i32 
    %1 = arith.muli %0, %c64_i32 : i32 
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32> 
    %3 = tt.splat %1 : i32 -> tensor<64xi32> 
    %4 = arith.addi %3, %2 : tensor<64xi32> 
    %5 = arith.mulf %arg5, %cst_4 : f32 
    %6 = tt.splat %5 : f32 -> tensor<64xf32> 
    %7 = tt.splat %5 : f32 -> tensor<64x64xf32> 
    %8 = arith.muli %0, %c64_i32 {tt.divisibility = dense<64> : tensor<1xi32>} : i32 
    %9 = arith.addi %0, %c1_i32 : i32 
    %10 = arith.muli %9, %c64_i32 : i32 
    %11 = tt.expand_dims %4 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32> 
    %12 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32> 
    %13 = tt.broadcast %11 : tensor<64x1xi32> -> tensor<64x64xi32> 
    %14 = tt.splat %5 : f32 -> tensor<64x64xf32> 
    %15 = tt.splat %5 : f32 -> tensor<64xf32> 
    scf.for %arg6 = %c0_i32 to %c128_i32 step %c1_i32  : i32 {
      %16 = arith.divsi %arg6, %c32_i32 : i32 
      %17 = arith.remsi %arg6, %c32_i32 : i32 
      %18 = arith.extsi %16 : i32 to i64 
      %19 = arith.muli %18, %c4194304_i64 : i64 
      %20 = arith.extsi %17 : i32 to i64 
      %21 = arith.muli %20, %c131072_i64 : i64 
      %22 = arith.addi %19, %21 : i64 
      %23 = tt.addptr %arg0, %22 : !tt.ptr<f16>, i64 
      %24 = tt.make_tensor_ptr %23, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%1, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16>> 
      // CHECK-NOT: annotation.mark %[[COPYDST0:.*]] {MayImplicitTransposeWithLastAxis} : memref<64x64xf16>
      // CHECK-NOT: annotation.mark %[[LOADED0:.*]] {MayImplicitTransposeWithLastAxis} : tensor<64x64xf16>
      %25 = tt.addptr %arg2, %22 : !tt.ptr<f16>, i64 
      %26 = tt.make_tensor_ptr %25, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16>> 
      // CHECK-NOT: annotation.mark %[[COPYDST1:.*]] {MayImplicitTransposeWithLastAxis} : memref<64x64xf16>
      // CHECK-NOT: annotation.mark %[[LOADED1:.*]] {MayImplicitTransposeWithLastAxis} : tensor<64x64xf16>
      %27 = tt.addptr %arg1, %22 : !tt.ptr<f16>, i64 
      %28 = tt.make_tensor_ptr %27, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16>> 
      // CHECK-NOT: annotation.mark %[[COPYDST2:.*]] {MayImplicitTransposeWithLastAxis} : memref<64x64xf16>
      // CHECK-NOT: annotation.mark %[[LOADED2:.*]] {MayImplicitTransposeWithLastAxis} : tensor<64x64xf16>
      %29 = tt.addptr %arg4, %22 : !tt.ptr<f16>, i64 
      %30 = tt.make_tensor_ptr %29, [%c2048_i64, %c64_i64], [%c64_i64, %c1_i64], [%1, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf16>> 
      // CHECK-NOT: annotation.mark %[[COPYDST3:.*]] {MayImplicitTransposeWithLastAxis} : memref<64x64xf16>
      // CHECK-NOT: annotation.mark %[[LOADED3:.*]] {MayImplicitTransposeWithLastAxis} : tensor<64x64xf16>
      %31 = tt.load %24 : !tt.ptr<tensor<64x64xf16>>
      %32:5 = scf.for %arg7 = %c0_i32 to %1 step %c64_i32 iter_args(%arg8 = %cst, %arg9 = %cst_3, %arg10 = %cst_0, %arg11 = %26, %arg12 = %28) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, !tt.ptr<tensor<64x64xf16>>, !tt.ptr<tensor<64x64xf16>>)  : i32 {
        %46 = tt.load %arg12 : !tt.ptr<tensor<64x64xf16>> 
        %47 = tt.trans %46 {order = array<i32: 1, 0>} : tensor<64x64xf16> -> tensor<64x64xf16> 
        %48 = tt.dot %31, %47, %cst_3 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32> 
        %49 = "tt.reduce"(%48) <{axis = 1 : i32}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %72 = arith.maxnumf %arg13, %arg14 : f32 
          tt.reduce.return %72 : f32 
        }) : (tensor<64x64xf32>) -> tensor<64xf32> 
        %50 = arith.mulf %49, %6 : tensor<64xf32> 
        %51 = arith.maxnumf %arg10, %50 : tensor<64xf32> 
        %52 = arith.mulf %48, %7 : tensor<64x64xf32> 
        %53 = tt.expand_dims %51 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32> 
        %54 = tt.broadcast %53 : tensor<64x1xf32> -> tensor<64x64xf32> 
        %55 = arith.subf %52, %54 : tensor<64x64xf32> 
        %56 = math.exp2 %55 : tensor<64x64xf32> 
        %57 = "tt.reduce"(%56) <{axis = 1 : i32}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %72 = arith.addf %arg13, %arg14 : f32 
          tt.reduce.return %72 : f32 
        }) : (tensor<64x64xf32>) -> tensor<64xf32> 
        %58 = arith.subf %arg10, %51 : tensor<64xf32> 
        %59 = math.exp2 %58 : tensor<64xf32> 
        %60 = arith.mulf %arg8, %59 : tensor<64xf32> 
        %61 = arith.addf %60, %57 : tensor<64xf32> 
        %62 = tt.expand_dims %59 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32> 
        %63 = tt.broadcast %62 : tensor<64x1xf32> -> tensor<64x64xf32> 
        %64 = arith.mulf %arg9, %63 : tensor<64x64xf32> 
        %65 = tt.load %arg11 : !tt.ptr<tensor<64x64xf16>> 
        %66 = arith.truncf %56 : tensor<64x64xf32> to tensor<64x64xf16> 
        %67 = tt.dot %66, %65, %64 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32> 
        %68 = arith.mulf %51, %6 : tensor<64xf32> 
        %69 = arith.divf %68, %cst_2 : tensor<64xf32> 
        %70 = tt.advance %arg11, [%c64_i32, %c0_i32] : <tensor<64x64xf16>> 
        %71 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x64xf16>> 
        scf.yield %61, %67, %69, %70, %71 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, !tt.ptr<tensor<64x64xf16>>, !tt.ptr<tensor<64x64xf16>> 
      } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>} 
      %33 = tt.advance %28, [%8, %c0_i32] : <tensor<64x64xf16>> 
      %34 = tt.advance %26, [%8, %c0_i32] : <tensor<64x64xf16>> 
      %35:5 = scf.for %arg7 = %8 to %10 step %c64_i32 iter_args(%arg8 = %32#0, %arg9 = %32#1, %arg10 = %32#2, %arg11 = %34, %arg12 = %33) -> (tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, !tt.ptr<tensor<64x64xf16>>, !tt.ptr<tensor<64x64xf16>>)  : i32 {
        %46 = tt.load %arg12 : !tt.ptr<tensor<64x64xf16>> 
        %47 = tt.trans %46 {order = array<i32: 1, 0>} : tensor<64x64xf16> -> tensor<64x64xf16> 
        %48 = tt.dot %31, %47, %cst_3 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32> 
        %49 = tt.splat %arg7 : i32 -> tensor<1x64xi32> 
        %50 = arith.addi %49, %12 : tensor<1x64xi32> 
        %51 = tt.broadcast %50 : tensor<1x64xi32> -> tensor<64x64xi32> 
        %52 = arith.cmpi sge, %13, %51 : tensor<64x64xi32> 
        %53 = arith.mulf %48, %14 : tensor<64x64xf32> 
        %54 = arith.select %52, %cst_3, %cst_1 : tensor<64x64xi1>, tensor<64x64xf32> 
        %55 = arith.addf %53, %54 : tensor<64x64xf32> 
        %56 = "tt.reduce"(%55) <{axis = 1 : i32}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %77 = arith.maxnumf %arg13, %arg14 : f32 
          tt.reduce.return %77 : f32 
        }) : (tensor<64x64xf32>) -> tensor<64xf32> 
        %57 = arith.maxnumf %arg10, %56 : tensor<64xf32> 
        %58 = tt.expand_dims %57 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32> 
        %59 = tt.broadcast %58 : tensor<64x1xf32> -> tensor<64x64xf32> 
        %60 = arith.subf %55, %59 : tensor<64x64xf32> 
        %61 = math.exp2 %60 : tensor<64x64xf32> 
        %62 = "tt.reduce"(%61) <{axis = 1 : i32}> ({
        ^bb0(%arg13: f32, %arg14: f32):
          %77 = arith.addf %arg13, %arg14 : f32 
          tt.reduce.return %77 : f32 
        }) : (tensor<64x64xf32>) -> tensor<64xf32> 
        %63 = arith.subf %arg10, %57 : tensor<64xf32> 
        %64 = math.exp2 %63 : tensor<64xf32> 
        %65 = arith.mulf %arg8, %64 : tensor<64xf32> 
        %66 = arith.addf %65, %62 : tensor<64xf32> 
        %67 = tt.expand_dims %64 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32> 
        %68 = tt.broadcast %67 : tensor<64x1xf32> -> tensor<64x64xf32> 
        %69 = arith.mulf %arg9, %68 : tensor<64x64xf32> 
        %70 = tt.load %arg11 : !tt.ptr<tensor<64x64xf16>> 
        %71 = arith.truncf %61 : tensor<64x64xf32> to tensor<64x64xf16> 
        %72 = tt.dot %71, %70, %69 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32> 
        %73 = arith.mulf %57, %15 : tensor<64xf32> 
        %74 = arith.divf %73, %cst_2 : tensor<64xf32> 
        %75 = tt.advance %arg11, [%c64_i32, %c0_i32] : <tensor<64x64xf16>> 
        %76 = tt.advance %arg12, [%c64_i32, %c0_i32] : <tensor<64x64xf16>> 
        scf.yield %66, %72, %74, %75, %76 : tensor<64xf32>, tensor<64x64xf32>, tensor<64xf32>, !tt.ptr<tensor<64x64xf16>>, !tt.ptr<tensor<64x64xf16>> 
      } {tt.divisibility_arg1 = dense<64> : tensor<1xi32>} 
      %36 = math.log2 %35#0 : tensor<64xf32> 
      %37 = arith.addf %35#2, %36 : tensor<64xf32> 
      %38 = tt.expand_dims %35#0 {axis = 1 : i32} : tensor<64xf32> -> tensor<64x1xf32> 
      %39 = tt.broadcast %38 : tensor<64x1xf32> -> tensor<64x64xf32> 
      %40 = arith.divf %35#1, %39 : tensor<64x64xf32> 
      %41 = arith.muli %arg6, %c2048_i32 : i32 
      %42 = tt.addptr %arg3, %41 : !tt.ptr<f32>, i32 
      %43 = tt.splat %42 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>> 
      %44 = tt.addptr %43, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32> 
      tt.store %44, %37 : tensor<64x!tt.ptr<f32>> 
      %45 = arith.truncf %40 : tensor<64x64xf32> to tensor<64x64xf16> 
      tt.store %30, %45 : !tt.ptr<tensor<64x64xf16>> 
    } 
    tt.return 
  } 
}

