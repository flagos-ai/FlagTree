// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(rewrite-assume-with-cuda-tile)" 2>&1 | FileCheck %s

// CHECK-NOT: error
module @kernel{
  tt.func public @assume_rewrite(
    %arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32},
    %arg3: i32 {tt.divisibility = 16 : i32},
    %arg4: i32 {tt.divisibility = 16 : i32},
    %arg5: i32 {tt.divisibility = 16 : i32},
    %arg6: i32 {tt.divisibility = 16 : i32},
    %arg7: i32 {tt.divisibility = 16 : i32},
    %arg8: i32 {tt.divisibility = 16 : i32},
    %arg9: i32 {tt.divisibility = 16 : i32},
    %arg10: i32 {tt.divisibility = 16 : i32},
    %arg11: !tt.ptr<i32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32>
    %c63_i32 = arith.constant 63 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %true = arith.constant true
    %c16_i64 = arith.constant 16 : i64
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i64 = arith.constant 1 : i64
    %0 = arith.extsi %arg6 : i32 to i64
    %1 = tt.make_tensor_descriptor %arg0, [%arg3, %arg5], [%0, %c1_i64] : <f16>, <tensor<64x64xf16>>
    %2 = arith.extsi %arg7 : i32 to i64
    %3 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%2, %c1_i64] : <f16>, <tensor<64x64xf16>>
    %4 = tt.get_program_id x : i32
    %5 = arith.addi %arg3, %c63_i32 : i32
    %6 = arith.divsi %5, %c64_i32 : i32
    %7 = arith.addi %arg4, %c63_i32 : i32
    %8 = arith.divsi %7, %c64_i32 : i32
    %9 = arith.addi %arg5, %c63_i32 : i32
    %10 = arith.divsi %9, %c64_i32 : i32
    %11 = arith.muli %6, %8 : i32
    %12 = tt.get_num_programs x : i32
    %13 = arith.muli %8, %c8_i32 : i32
    %14 = tt.ptr_to_int %arg2 : !tt.ptr<f16> -> i64
    %15 = arith.remsi %14, %c16_i64 : i64
    %16 = arith.cmpi eq, %15, %c0_i64 : i64
    %17 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f16> to !cuda_tile.ptr<f16>
    %18 = builtin.unrealized_conversion_cast %arg8 : i32 to !cuda_tile.tile<i32>
    %19 = builtin.unrealized_conversion_cast %arg3 : i32 to !cuda_tile.tile<i32>
    %20 = builtin.unrealized_conversion_cast %arg4 : i32 to !cuda_tile.tile<i32>
    %21 = builtin.unrealized_conversion_cast %17 : !cuda_tile.ptr<f16> to !cuda_tile.tile<ptr<f16>>
    %22 = cuda_tile.make_token : token
    %23 = builtin.unrealized_conversion_cast %c0_i32 : i32 to !cuda_tile.tile<i32>
    %24 = builtin.unrealized_conversion_cast %true : i1 to !cuda_tile.tile<i1>
    scf.for %arg12 = %4 to %11 step %12  : i32 {
      %25 = arith.divsi %arg12, %13 : i32
      %26 = arith.muli %25, %c8_i32 : i32
      %27 = arith.subi %6, %26 : i32
      %28 = arith.minsi %27, %c8_i32 : i32
      %29 = arith.remsi %arg12, %28 : i32
      %30 = arith.addi %26, %29 : i32
      %31 = arith.remsi %arg12, %13 : i32
      %32 = arith.divsi %31, %28 : i32
      %33 = arith.muli %30, %c64_i32 : i32
      %34 = arith.muli %32, %c64_i32 : i32
      %35 = scf.for %arg13 = %c0_i32 to %10 step %c1_i32 iter_args(%arg14 = %cst) -> (tensor<64x64xf32>)  : i32 {
        %43 = arith.muli %arg13, %c64_i32 : i32
        %44 = tt.descriptor_load %1[%33, %43] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
        %45 = tt.descriptor_load %3[%34, %43] : !tt.tensordesc<tensor<64x64xf16>> -> tensor<64x64xf16>
        %46 = tt.trans %45 {order = array<i32: 1, 0>} : tensor<64x64xf16> -> tensor<64x64xf16>
        %47 = tt.dot %44, %46, %arg14, inputPrecision = tf32 : tensor<64x64xf16> * tensor<64x64xf16> -> tensor<64x64xf32>
        scf.yield %47 : tensor<64x64xf32>
      }
      llvm.intr.assume %16 : i1
      %tview = cuda_tile.make_tensor_view %21, shape = [%19, %20], strides = [%18, 1] : tile<i32> -> tensor_view<?x?xf16, strides=[?,1]>
      %pview = cuda_tile.make_partition_view %tview : partition_view<tile=(64x64), padding_value = zero, tensor_view<?x?xf16, strides=[?,1]>>
      %36 = tt.addptr %arg11, %arg12 : !tt.ptr<i32>, i32
      %37 = arith.truncf %35 : tensor<64x64xf32> to tensor<64x64xf16>
      %38 = builtin.unrealized_conversion_cast %37 : tensor<64x64xf16> to !cuda_tile.tile<64x64xf16>
      %39 = builtin.unrealized_conversion_cast %30 : i32 to !cuda_tile.tile<i32>
      %40 = builtin.unrealized_conversion_cast %32 : i32 to !cuda_tile.tile<i32>
      %41 = cuda_tile.store_view_tko release device %38, %pview[%39, %40] token = %22 : tile<64x64xf16>, partition_view<tile=(64x64), padding_value = zero, tensor_view<?x?xf16, strides=[?,1]>>, tile<i32> -> token
      %42 = builtin.unrealized_conversion_cast %36 : !tt.ptr<i32> to !cuda_tile.tile<ptr<i32>>
      %result, %result_token = cuda_tile.atomic_rmw_tko release device %42, xchg, %23, %24 token=%41 : tile<ptr<i32>>, tile<i32>, tile<i1> -> tile<i32>, token
    } {tt.flatten}
    tt.return
  }
}
