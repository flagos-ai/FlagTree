// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func public @scalar_mask_loop(%arg0: !tt.ptr<f8E4M3FN> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<4.480000e+02> : tensor<1xf32>
    %cst_0 = arith.constant dense<-4.480000e+02> : tensor<1xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.get_num_programs x : i32
    %2 = arith.addi %arg3, %1 : i32
    %3 = arith.subi %2, %c1_i32 : i32
    %4 = arith.divsi %3, %1 : i32
    %5 = tt.load %arg2 : !tt.ptr<f32>
    %6 = tt.splat %0 : i32 -> tensor<1xi32>
    %7 = scf.for %arg4 = %c0_i32 to %4 step %c1_i32 iter_args(%arg5 = %6) -> (tensor<1xi32>)  : i32 {
      %8 = tt.splat %arg3 : i32 -> tensor<1xi32>
      %9 = arith.cmpi slt, %arg5, %8 : tensor<1xi32>
      %10 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1x!tt.ptr<bf16>>
      %11 = tt.addptr %10, %arg5 : tensor<1x!tt.ptr<bf16>>, tensor<1xi32>
      %12 = tt.load %11, %9 : tensor<1x!tt.ptr<bf16>>
      %13 = arith.extf %12 : tensor<1xbf16> to tensor<1xf32>
      %14 = tt.splat %5 : f32 -> tensor<1xf32>
      %15 = arith.mulf %13, %14 : tensor<1xf32>
      %16 = tt.clampf %15, %cst_0, %cst, propagateNan = none : tensor<1xf32>
      %17 = tt.fp_to_fp %16, rounding = rtne : tensor<1xf32> -> tensor<1xf8E4M3FN>
      %18 = tt.splat %arg0 : !tt.ptr<f8E4M3FN> -> tensor<1x!tt.ptr<f8E4M3FN>>
      %19 = tt.addptr %18, %arg5 : tensor<1x!tt.ptr<f8E4M3FN>>, tensor<1xi32>
      tt.store %19, %17, %9 : tensor<1x!tt.ptr<f8E4M3FN>>
      %20 = tt.splat %1 : i32 -> tensor<1xi32>
      %21 = arith.addi %arg5, %20 : tensor<1xi32>
      scf.yield %21 : tensor<1xi32>
    }
    tt.return
  }
}


// CHECK: %8 = scf.for %arg4 = %c0_i32 to %6 step %c1_i32 iter_args(%arg5 = %1) -> (index)  : i32 {
// CHECK:   %9 = tts.make_tptr %arg1 to sizes: [1], strides: [%c0], offsets: [%arg5], shape: [0], order: [] : <bf16> to tensor<1x!tt.ptr<bf16>>
// CHECK:   %10 = "tts.load"(%9) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64: 1>}> : (tensor<1x!tt.ptr<bf16>>) -> tensor<1xbf16>