// RUN: triton-shared-opt --triton-to-linalg-experimental --split-input-file %s | FileCheck %s

// Make sure no tts.get_structured_state in the end.
// CHECK-LABLE: tt.func public @_moe_linear_kernel
// CHECK-NOT: tts.get_structured_state

module attributes {maia.triton_kernel} {
  tt.func public @_moe_linear_kernel(%arg0: !tt.ptr<f32> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {maia.rank = 3 : i32, tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg3: !tt.ptr<f32> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg4: !tt.ptr<i32> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg5: !tt.ptr<i32> {maia.rank = 2 : i32, tt.divisibility = 16 : i32}, %arg6: !tt.ptr<i32> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}, %arg7: !tt.ptr<i32> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}, %arg8: !tt.ptr<i32> {maia.rank = 1 : i32, tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}, %arg12: i32 {tt.divisibility = 16 : i32}, %arg13: i32 {tt.divisibility = 16 : i32}, %arg14: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %c63_i32 = arith.constant 63 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<64> : tensor<64x64xi32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %cst_1 = arith.constant dense<128> : tensor<64xi32>
    %c128_i32 = arith.constant 128 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg14, %c63_i32 : i32
    %2 = arith.divsi %1, %c64_i32 : i32
    %3 = arith.divsi %0, %c16_i32 : i32
    %4 = arith.muli %3, %c8_i32 : i32
    %5 = arith.subi %2, %4 : i32
    %6 = arith.minsi %5, %c8_i32 : i32
    %7 = arith.remsi %0, %6 : i32
    %8 = arith.addi %4, %7 : i32
    %9 = arith.remsi %0, %c16_i32 : i32
    %10 = arith.divsi %9, %6 : i32
    %11 = tt.addptr %arg7, %8 : !tt.ptr<i32>, i32
    %12 = tt.load %11 : !tt.ptr<i32>
    %13 = arith.cmpi slt, %12, %c0_i32 : i32
    cf.cond_br %13, ^bb1, ^bb2
  ^bb1:  // 2 preds: ^bb0, ^bb2
    tt.return
  ^bb2:  // pred: ^bb0
    %14 = tt.addptr %arg8, %8 : !tt.ptr<i32>, i32
    %15 = tt.load %14 : !tt.ptr<i32>
    %16 = tt.addptr %arg6, %12 : !tt.ptr<i32>, i32
    %17 = tt.load %16 : !tt.ptr<i32>
    %18 = tt.addptr %arg4, %12 : !tt.ptr<i32>, i32
    %19 = tt.addptr %arg5, %12 : !tt.ptr<i32>, i32
    %20 = arith.muli %12, %arg10 : i32
    %21 = tt.addptr %arg1, %20 : !tt.ptr<f32>, i32
    %22 = arith.cmpi sge, %15, %17 : i32
    cf.cond_br %22, ^bb1, ^bb3
  ^bb3:  // pred: ^bb2
    %23 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %24 = tt.splat %15 : i32 -> tensor<64xi32>
    %25 = arith.addi %24, %23 : tensor<64xi32>
    %26 = arith.muli %10, %c64_i32 : i32
    %27 = tt.splat %26 : i32 -> tensor<64xi32>
    %28 = arith.addi %27, %23 : tensor<64xi32>
    %29 = arith.remsi %28, %cst_1 : tensor<64xi32>
    %30 = tt.splat %17 : i32 -> tensor<64xi32>
    %31 = arith.cmpi slt, %25, %30 : tensor<64xi32>
    %32 = tt.splat %18 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
    %33 = tt.addptr %32, %25 : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
    %34 = tt.load %33, %31 : tensor<64x!tt.ptr<i32>>
    %35 = tt.expand_dims %34 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %36 = tt.splat %arg9 : i32 -> tensor<64x1xi32>
    %37 = arith.muli %35, %36 : tensor<64x1xi32>
    %38 = tt.expand_dims %23 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %39 = tt.broadcast %37 : tensor<64x1xi32> -> tensor<64x64xi32>
    %40 = tt.broadcast %38 : tensor<1x64xi32> -> tensor<64x64xi32>
    %41 = arith.addi %39, %40 : tensor<64x64xi32>
    %42 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %43 = tt.addptr %42, %41 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %44 = tt.expand_dims %23 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %45 = tt.expand_dims %29 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %46 = tt.splat %arg11 : i32 -> tensor<1x64xi32>
    %47 = arith.muli %45, %46 : tensor<1x64xi32>
    %48 = tt.broadcast %44 : tensor<64x1xi32> -> tensor<64x64xi32>
    %49 = tt.broadcast %47 : tensor<1x64xi32> -> tensor<64x64xi32>
    %50 = arith.addi %48, %49 : tensor<64x64xi32>
    %51 = tt.splat %21 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %52 = tt.addptr %51, %50 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %53:3 = scf.for %arg15 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg16 = %cst_0, %arg17 = %43, %arg18 = %52) -> (tensor<64x64xf32>, tensor<64x64x!tt.ptr<f32>>, tensor<64x64x!tt.ptr<f32>>)  : i32 {
      %79 = arith.muli %arg15, %c64_i32 : i32
      %80 = arith.subi %c128_i32, %79 : i32
      %81 = tt.expand_dims %31 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
      %82 = tt.splat %80 : i32 -> tensor<1x64xi32>
      %83 = arith.cmpi slt, %38, %82 : tensor<1x64xi32>
      %84 = tt.broadcast %81 : tensor<64x1xi1> -> tensor<64x64xi1>
      %85 = tt.broadcast %83 : tensor<1x64xi1> -> tensor<64x64xi1>
      %86 = arith.andi %84, %85 : tensor<64x64xi1>
      %87 = tt.load %arg17, %86, %cst_0 : tensor<64x64x!tt.ptr<f32>>
      %88 = tt.splat %80 : i32 -> tensor<64x1xi32>
      %89 = arith.cmpi slt, %44, %88 : tensor<64x1xi32>
      %90 = tt.broadcast %89 : tensor<64x1xi1> -> tensor<64x64xi1>
      %91 = tt.load %arg18, %90, %cst_0 : tensor<64x64x!tt.ptr<f32>>
      %92 = tt.dot %87, %91, %arg16 : tensor<64x64xf32> * tensor<64x64xf32> -> tensor<64x64xf32>
      %93 = tt.addptr %arg17, %cst : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
      %94 = tt.addptr %arg18, %cst : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
      scf.yield %92, %93, %94 : tensor<64x64xf32>, tensor<64x64x!tt.ptr<f32>>, tensor<64x64x!tt.ptr<f32>>
    }
    %54 = arith.muli %12, %arg12 : i32
    %55 = tt.addptr %arg2, %54 : !tt.ptr<f32>, i32
    %56 = tt.splat %55 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %57 = tt.addptr %56, %45 : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %58 = tt.load %57 : tensor<1x64x!tt.ptr<f32>>
    %59 = tt.broadcast %58 : tensor<1x64xf32> -> tensor<64x64xf32>
    %60 = arith.addf %53#0, %59 : tensor<64x64xf32>
    %61 = arith.cmpi slt, %28, %cst_1 : tensor<64xi32>
    %62 = tt.splat %19 : !tt.ptr<i32> -> tensor<64x!tt.ptr<i32>>
    %63 = tt.addptr %62, %25 : tensor<64x!tt.ptr<i32>>, tensor<64xi32>
    %64 = tt.load %63, %31 : tensor<64x!tt.ptr<i32>>
    %65 = tt.expand_dims %64 {axis = 1 : i32} : tensor<64xi32> -> tensor<64x1xi32>
    %66 = tt.splat %arg13 : i32 -> tensor<64x1xi32>
    %67 = arith.muli %65, %66 : tensor<64x1xi32>
    %68 = tt.expand_dims %28 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %69 = tt.broadcast %67 : tensor<64x1xi32> -> tensor<64x64xi32>
    %70 = tt.broadcast %68 : tensor<1x64xi32> -> tensor<64x64xi32>
    %71 = arith.addi %69, %70 : tensor<64x64xi32>
    %72 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<64x64x!tt.ptr<f32>>
    %73 = tt.addptr %72, %71 : tensor<64x64x!tt.ptr<f32>>, tensor<64x64xi32>
    %74 = tt.expand_dims %31 {axis = 1 : i32} : tensor<64xi1> -> tensor<64x1xi1>
    %75 = tt.expand_dims %61 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %76 = tt.broadcast %74 : tensor<64x1xi1> -> tensor<64x64xi1>
    %77 = tt.broadcast %75 : tensor<1x64xi1> -> tensor<64x64xi1>
    %78 = arith.andi %76, %77 : tensor<64x64xi1>
    tt.store %73, %60, %78 : tensor<64x64x!tt.ptr<f32>>
    tt.return
  }
}
