// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @triton_divRn(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %c64_i32 = arith.constant 64 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c2048_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %5 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    scf.for %arg3 = %c0_i32 to %c32_i32 step %c1_i32  : i32 {
      %6 = arith.muli %arg3, %c64_i32 : i32
      %7 = arith.addi %1, %6 : i32
      %8 = tt.splat %7 : i32 -> tensor<64xi32>
      %9 = arith.addi %8, %2 : tensor<64xi32>
      %10 = tt.addptr %3, %9 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %11 = tt.load %10 : tensor<64x!tt.ptr<f32>>
      %12 = tt.addptr %4, %9 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %13 = tt.load %12 : tensor<64x!tt.ptr<f32>>
      %14 = tt.precise_divf %11, %13 : tensor<64xf32>
      %15 = tt.addptr %5, %9 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %15, %14 : tensor<64x!tt.ptr<f32>>
    }
    tt.return
  }
}

//CHECK: %[[VAL0:.*]] = bufferization.to_tensor %alloc restrict writable : memref<64xf32>
//CHECK: %[[VAL1:.*]] = bufferization.to_tensor %alloc_1 restrict writable : memref<64xf32>
//CHECK: %[[VAL2:.*]] = arith.divf %in, %in_3 : f32


