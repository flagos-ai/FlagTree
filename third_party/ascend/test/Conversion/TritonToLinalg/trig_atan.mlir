// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s


module {
  tt.func public @triton_atan(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<64xi32>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %5 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    scf.for %arg4 = %c0_i32 to %c128_i32 step %c64_i32  : i32 {
      %6 = arith.addi %1, %arg4 : i32
      %7 = tt.splat %6 : i32 -> tensor<64xi32>
      %8 = arith.addi %7, %2 : tensor<64xi32>
      %9 = arith.cmpi slt, %8, %3 : tensor<64xi32>
      %10 = tt.addptr %4, %8 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %11 = tt.load %10, %9 : tensor<64x!tt.ptr<f32>>
      %12 = tt.extern_elementwise %11 {libname = "", libpath = "", pure = true, symbol = "__hmf_atanf"} : (tensor<64xf32>) -> tensor<64xf32>
      %13 = tt.addptr %5, %8 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %13, %12, %9 : tensor<64x!tt.ptr<f32>>
    }
    tt.return
  }
}


//CHECK: %mapped = linalg.map { func.call {callee = @__hmf_atanf} }
