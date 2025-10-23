// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @triton_tan(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c32768_i32 : i32
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %3 = tt.splat %arg3 : i32 -> tensor<1024xi32>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %5 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    scf.for %arg4 = %c0_i32 to %c32768_i32 step %c1024_i32  : i32 {
      %6 = arith.addi %1, %arg4 : i32
      %7 = tt.splat %6 : i32 -> tensor<1024xi32>
      %8 = arith.addi %7, %2 : tensor<1024xi32>
      %9 = arith.cmpi slt, %8, %3 : tensor<1024xi32>
      %10 = tt.addptr %4, %8 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      %11 = tt.load %10, %9 : tensor<1024x!tt.ptr<f32>>
      %12 = tt.extern_elementwise %11 {libname = "", libpath = "", pure = true, symbol = "__hmf_tanf"} : (tensor<1024xf32>) -> tensor<1024xf32>
      %13 = tt.addptr %5, %8 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
      tt.store %13, %12, %9 : tensor<1024x!tt.ptr<f32>>
    }
    tt.return
  }
}

//CHECK: %mapped = linalg.map { func.call {callee = @__hmf_tanf} }
