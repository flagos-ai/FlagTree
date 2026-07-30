// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func public @mask_loop(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %2 = tt.get_num_programs x : i32
    %3 = arith.muli %2, %c2_i32 : i32
    %4 = arith.addi %arg3, %c3_i32 : i32
    %5 = arith.divsi %4, %c4_i32 : i32
    %6 = arith.muli %0, %c2_i32 : i32
    %7 = tt.splat %6 : i32 -> tensor<2xi32>
    %8 = arith.addi %7, %1 : tensor<2xi32>
    %9 = tt.splat %arg3 : i32 -> tensor<2xi32>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
    %11 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
    %12 = tt.splat %3 : i32 -> tensor<2xi32>
    %13 = scf.for %arg4 = %c0_i32 to %5 step %c1_i32 iter_args(%arg5 = %8) -> (tensor<2xi32>)  : i32 {
      %14 = arith.cmpi slt, %arg5, %9 : tensor<2xi32>
      %15 = tt.addptr %10, %arg5 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
      %16 = tt.load %15, %14 : tensor<2x!tt.ptr<f32>>
      %17 = tt.addptr %11, %arg5 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
      tt.store %17, %16, %14 : tensor<2x!tt.ptr<f32>>
      %18 = arith.addi %arg5, %12 : tensor<2xi32>
      scf.yield %18 : tensor<2xi32>
    }
    tt.return
  }
}

// CHECK: %8 = scf.for %arg4 = %c0_i32 to %5 step %c1_i32 iter_args(%arg5 = %7) -> (index)  : i32 {
// CHECK:     %9 = tts.make_tptr %arg1
// CHECK:     %10 = arith.addi %arg5, %c2 : index
// CHECK:     %11 = arith.index_cast %arg3 : i32 to index
// CHECK:     %12 = arith.minsi %10, %11 : index
// CHECK:     %13 = arith.maxsi %12, %arg5 : index
// CHECK:     %14 = arith.subi %13, %arg5 : index
// CHECK:     %15 = "tts.load"(%9, %14)
