// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
tt.func public @fn_npu_flip(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<4x8x8xi32>
    %cst_0 = arith.constant dense<8> : tensor<1x8x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<4x1x1xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %2 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %3 = tt.expand_dims %2 {axis = 2 : i32} : tensor<4x1xi32> -> tensor<4x1x1xi32>
    %4 = arith.muli %3, %cst_1 : tensor<4x1x1xi32>
    %5 = arith.muli %4, %cst_1 : tensor<4x1x1xi32>
    %6 = tt.expand_dims %1 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %8 = arith.muli %7, %cst_0 : tensor<1x8x1xi32>
    %9 = tt.broadcast %5 : tensor<4x1x1xi32> -> tensor<4x8x1xi32>
    %10 = tt.broadcast %8 : tensor<1x8x1xi32> -> tensor<4x8x1xi32>
    %11 = arith.addi %9, %10 : tensor<4x8x1xi32>
    %12 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32>
    %13 = tt.broadcast %11 : tensor<4x8x1xi32> -> tensor<4x8x8xi32>
    %14 = tt.broadcast %12 : tensor<1x1x8xi32> -> tensor<4x8x8xi32>
    %15 = arith.addi %13, %14 : tensor<4x8x8xi32>
    %16 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x8x8x!tt.ptr<f32>>
    %17 = tt.addptr %16, %15 : tensor<4x8x8x!tt.ptr<f32>>, tensor<4x8x8xi32>
    %18 = tt.load %17 : tensor<4x8x8x!tt.ptr<f32>>
    %19 = tt.extern_elementwise %18, %cst {libname = "", libpath = "", pure = true, symbol = "__hmf_flipf"} : (tensor<4x8x8xf32>, tensor<4x8x8xi32>) -> tensor<4x8x8xf32>
    %20 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x8x8x!tt.ptr<f32>>
    %21 = tt.addptr %20, %15 : tensor<4x8x8x!tt.ptr<f32>>, tensor<4x8x8xi32>
    tt.store %21, %19 : tensor<4x8x8x!tt.ptr<f32>>
    tt.return
}
}

//CHECK: func.func private @__hmf_flipf(f32, i32) -> f32 attributes {llvm.readnone}
//CHECK-NOT: tt.extern_elementwise
//CHECK: %[[RESULT:.*]] = linalg.map { func.call {callee = @__hmf_flipf} } ins(%[[TENSOR:.*]], %[[DIM:.*]] : tensor<4x8x8xf32>, tensor<4x8x8xi32>) outs(%[[TENSOR]] : tensor<4x8x8xf32>)
