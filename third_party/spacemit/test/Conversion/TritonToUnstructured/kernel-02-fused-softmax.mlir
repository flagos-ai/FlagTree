// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

module {
  tt.func public @softmax_kernel_012345(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant 0xFF800000 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %6 = tt.splat %arg4 : i32 -> tensor<128xi32>
    %7 = arith.cmpi slt, %3, %6 : tensor<128xi32>
    %8 = tt.splat %cst : f32 -> tensor<128xf32>
    %9 = tt.load %5, %7, %8 : tensor<128x!tt.ptr<f32>>
    %10 = "tt.reduce"(%9) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.cmpf ogt, %arg5, %arg6 : f32
      %22 = arith.select %21, %arg5, %arg6 : f32
      tt.reduce.return %22 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    %11 = tt.splat %10 : f32 -> tensor<128xf32>
    %12 = arith.subf %9, %11 : tensor<128xf32>
    %13 = math.exp %12 : tensor<128xf32>
    %14 = "tt.reduce"(%13) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.addf %arg5, %arg6 : f32
      tt.reduce.return %21 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    %15 = tt.splat %14 : f32 -> tensor<128xf32>
    %16 = arith.divf %13, %15 : tensor<128xf32>
    %17 = arith.muli %0, %arg3 : i32
    %18 = tt.addptr %arg0, %17 : !tt.ptr<f32>, i32
    %19 = tt.splat %18 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %20 = tt.addptr %19, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %20, %16, %7 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-NOT: tt.addptr
// CHECK-NOT: tt.load
// CHECK-NOT: tt.store

// CHECK-COUNT-1: tts.gather %arg{{[0-9]+}}
// CHECK-NOT: tts.gather %arg{{[0-9]+}}
// CHECK-COUNT-1: tts.scatter {{.+}} into %arg{{[0-9]+}}
// CHECK-NOT:    tts.scatter {{.+}} into %arg{{[0-9]+}}
