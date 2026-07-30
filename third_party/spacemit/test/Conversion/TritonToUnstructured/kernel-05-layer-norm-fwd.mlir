// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @_layer_norm_fwd_fused_0123456789(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>, %arg5: !tt.ptr<f32>, %arg6: i32, %arg7: i32, %arg8: f32) {
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg6 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.addptr %arg0, %1 : !tt.ptr<f32>, i32
    %4 = tt.splat %cst_0 : f32 -> tensor<256xf32>
    %5 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %6 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %7 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg10 = %4) -> (tensor<256xf32>)  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %5 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %6 : tensor<256xi32>
      %35 = tt.addptr %7, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34, %4 : tensor<256x!tt.ptr<f32>>
      %37 = arith.addf %arg10, %36 : tensor<256xf32>
      scf.yield %37 : tensor<256xf32>
    }
    %9 = "tt.reduce"(%8) ({
    ^bb0(%arg9: f32, %arg10: f32):
      %32 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %32 : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %10 = arith.sitofp %arg7 : i32 to f32
    %11 = arith.divf %9, %10 : f32
    %12 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %13 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %14 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %15 = tt.splat %11 : f32 -> tensor<256xf32>
    %16 = scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32 iter_args(%arg10 = %4) -> (tensor<256xf32>)  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %12 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %13 : tensor<256xi32>
      %35 = tt.addptr %14, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34, %4 : tensor<256x!tt.ptr<f32>>
      %37 = arith.subf %36, %15 : tensor<256xf32>
      %38 = arith.select %34, %37, %4 : tensor<256xi1>, tensor<256xf32>
      %39 = arith.mulf %38, %38 : tensor<256xf32>
      %40 = arith.addf %arg10, %39 : tensor<256xf32>
      scf.yield %40 : tensor<256xf32>
    }
    %17 = "tt.reduce"(%16) ({
    ^bb0(%arg9: f32, %arg10: f32):
      %32 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %32 : f32
    }) {axis = 0 : i32} : (tensor<256xf32>) -> f32
    %18 = arith.divf %17, %10 : f32
    %19 = arith.addf %18, %arg8 : f32
    %20 = math.sqrt %19 : f32
    %21 = arith.divf %cst, %20 : f32
    %22 = tt.addptr %arg4, %0 : !tt.ptr<f32>, i32
    tt.store %22, %11 : !tt.ptr<f32>
    %23 = tt.addptr %arg5, %0 : !tt.ptr<f32>, i32
    tt.store %23, %21 : !tt.ptr<f32>
    %24 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %25 = tt.splat %arg7 : i32 -> tensor<256xi32>
    %26 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %27 = tt.splat %arg3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %28 = tt.splat %3 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %29 = tt.splat %11 : f32 -> tensor<256xf32>
    %30 = tt.splat %21 : f32 -> tensor<256xf32>
    %31 = tt.splat %2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    scf.for %arg9 = %c0_i32 to %arg7 step %c256_i32  : i32 {
      %32 = tt.splat %arg9 : i32 -> tensor<256xi32>
      %33 = arith.addi %32, %24 : tensor<256xi32>
      %34 = arith.cmpi slt, %33, %25 : tensor<256xi32>
      %35 = tt.addptr %26, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %36 = tt.load %35, %34 : tensor<256x!tt.ptr<f32>>
      %37 = tt.addptr %27, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %38 = tt.load %37, %34 : tensor<256x!tt.ptr<f32>>
      %39 = tt.addptr %28, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      %40 = tt.load %39, %34, %4 : tensor<256x!tt.ptr<f32>>
      %41 = arith.subf %40, %29 : tensor<256xf32>
      %42 = arith.mulf %41, %30 : tensor<256xf32>
      %43 = arith.mulf %42, %36 : tensor<256xf32>
      %44 = arith.addf %43, %38 : tensor<256xf32>
      %45 = tt.addptr %31, %33 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
      tt.store %45, %44, %34 : tensor<256x!tt.ptr<f32>>
    }
    tt.return
  }
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store

// CHECK-COUNT-5: tts.gather %arg{{[0-9]+}}
// CHECK-NOT: tts.gather %arg{{[0-9]+}}
