// RUN: triton-adapter-opt --triton-linearize '--discrete-mask-access-conversion=compile-on-a5=True force_simt_template=False' --triton-to-annotation '--triton-to-unstructure=compile-on-a5=True force_simt_template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-a5=True' --split-input-file %s | FileCheck %s

module {
		tt.func public @fn_npu_(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}, %arg3: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<4> : tensor<1x29x1xi32>
		%cst_0 = arith.constant dense<4> : tensor<2x1x1xi32>
		%cst_1 = arith.constant dense<29> : tensor<2x1x1xi32>
		%c4_i32 = arith.constant 4 : i32
		%c29_i32 = arith.constant 29 : i32
		%c2_i32 = arith.constant 2 : i32
		%0 = tt.get_program_id x : i32
		%1 = arith.muli %0, %c2_i32 : i32
		%2 = tt.get_program_id y : i32
		%3 = arith.muli %2, %c29_i32 : i32
		%4 = tt.get_program_id z : i32
		%5 = arith.muli %4, %c4_i32 : i32
		%6 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
		%7 = tt.splat %1 : i32 -> tensor<2xi32>
		%8 = arith.addi %6, %7 : tensor<2xi32>
		%9 = tt.make_range {end = 29 : i32, start = 0 : i32} : tensor<29xi32>
		%10 = tt.splat %3 : i32 -> tensor<29xi32>
		%11 = arith.addi %9, %10 : tensor<29xi32>
		%12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
		%13 = tt.splat %5 : i32 -> tensor<4xi32>
		%14 = arith.addi %12, %13 : tensor<4xi32>
		%15 = tt.expand_dims %8 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
		%16 = tt.expand_dims %15 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
		%17 = arith.muli %16, %cst_1 : tensor<2x1x1xi32>
		%18 = arith.muli %17, %cst_0 : tensor<2x1x1xi32>
		%19 = tt.expand_dims %11 {axis = 0 : i32} : tensor<29xi32> -> tensor<1x29xi32>
		%20 = tt.expand_dims %19 {axis = 2 : i32} : tensor<1x29xi32> -> tensor<1x29x1xi32>
		%21 = arith.muli %20, %cst : tensor<1x29x1xi32>
		%22 = tt.broadcast %18 : tensor<2x1x1xi32> -> tensor<2x29x1xi32>
		%23 = tt.broadcast %21 : tensor<1x29x1xi32> -> tensor<2x29x1xi32>
		%24 = arith.addi %22, %23 : tensor<2x29x1xi32>
		%25 = tt.expand_dims %14 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
		%26 = tt.expand_dims %25 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
		%27 = tt.broadcast %24 : tensor<2x29x1xi32> -> tensor<2x29x4xi32>
		%28 = tt.broadcast %26 : tensor<1x1x4xi32> -> tensor<2x29x4xi32>
		%29 = arith.addi %27, %28 : tensor<2x29x4xi32>
		%30 = tt.splat %arg1 : !tt.ptr -> tensor<2x29x4x!tt.ptr>
		%31 = tt.addptr %30, %29 : tensor<2x29x4x!tt.ptr>, tensor<2x29x4xi32>
		%32 = tt.load %31 : tensor<2x29x4x!tt.ptr>
		%33 = math.exp2 %32 : tensor<2x29x4xf8E5M2>
		%34 = tt.splat %arg0 : !tt.ptr -> tensor<2x29x4x!tt.ptr>
		%35 = tt.addptr %34, %29 : tensor<2x29x4x!tt.ptr>, tensor<2x29x4xi32>
		tt.store %35, %33 : tensor<2x29x4x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[ALLOC_INPUT:[A-Za-z0-9_]+]] = memref.alloc() : memref<2x29x4xf8E5M2>
// CHECK: %[[TENSOR_INPUT:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC_INPUT]] restrict writable : memref<2x29x4xf8E5M2>

// CHECK: %[[EXP2_RESULT:[A-Za-z0-9_]+]] = math.exp2 %[[TENSOR_INPUT]] : tensor<2x29x4xf8E5M2>

// -----

module {
		tt.func public @fn_npu_(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}, %arg3: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<4> : tensor<1x29x1xi32>
		%cst_0 = arith.constant dense<4> : tensor<2x1x1xi32>
		%cst_1 = arith.constant dense<29> : tensor<2x1x1xi32>
		%c4_i32 = arith.constant 4 : i32
		%c29_i32 = arith.constant 29 : i32
		%c2_i32 = arith.constant 2 : i32
		%0 = tt.get_program_id x : i32
		%1 = arith.muli %0, %c2_i32 : i32
		%2 = tt.get_program_id y : i32
		%3 = arith.muli %2, %c29_i32 : i32
		%4 = tt.get_program_id z : i32
		%5 = arith.muli %4, %c4_i32 : i32
		%6 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
		%7 = tt.splat %1 : i32 -> tensor<2xi32>
		%8 = arith.addi %6, %7 : tensor<2xi32>
		%9 = tt.make_range {end = 29 : i32, start = 0 : i32} : tensor<29xi32>
		%10 = tt.splat %3 : i32 -> tensor<29xi32>
		%11 = arith.addi %9, %10 : tensor<29xi32>
		%12 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
		%13 = tt.splat %5 : i32 -> tensor<4xi32>
		%14 = arith.addi %12, %13 : tensor<4xi32>
		%15 = tt.expand_dims %8 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
		%16 = tt.expand_dims %15 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
		%17 = arith.muli %16, %cst_1 : tensor<2x1x1xi32>
		%18 = arith.muli %17, %cst_0 : tensor<2x1x1xi32>
		%19 = tt.expand_dims %11 {axis = 0 : i32} : tensor<29xi32> -> tensor<1x29xi32>
		%20 = tt.expand_dims %19 {axis = 2 : i32} : tensor<1x29xi32> -> tensor<1x29x1xi32>
		%21 = arith.muli %20, %cst : tensor<1x29x1xi32>
		%22 = tt.broadcast %18 : tensor<2x1x1xi32> -> tensor<2x29x1xi32>
		%23 = tt.broadcast %21 : tensor<1x29x1xi32> -> tensor<2x29x1xi32>
		%24 = arith.addi %22, %23 : tensor<2x29x1xi32>
		%25 = tt.expand_dims %14 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
		%26 = tt.expand_dims %25 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
		%27 = tt.broadcast %24 : tensor<2x29x1xi32> -> tensor<2x29x4xi32>
		%28 = tt.broadcast %26 : tensor<1x1x4xi32> -> tensor<2x29x4xi32>
		%29 = arith.addi %27, %28 : tensor<2x29x4xi32>
		%30 = tt.splat %arg1 : !tt.ptr -> tensor<2x29x4x!tt.ptr>
		%31 = tt.addptr %30, %29 : tensor<2x29x4x!tt.ptr>, tensor<2x29x4xi32>
		%32 = tt.load %31 : tensor<2x29x4x!tt.ptr>
		%33 = math.exp2 %32 : tensor<2x29x4xf8E4M3FN>
		%34 = tt.splat %arg0 : !tt.ptr -> tensor<2x29x4x!tt.ptr>
		%35 = tt.addptr %34, %29 : tensor<2x29x4x!tt.ptr>, tensor<2x29x4xi32>
		tt.store %35, %33 : tensor<2x29x4x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[ALLOC_INPUT:[A-Za-z0-9_]+]] = memref.alloc() : memref<2x29x4xf8E4M3FN>
// CHECK: %[[TENSOR_INPUT:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC_INPUT]] restrict writable : memref<2x29x4xf8E4M3FN>

// CHECK: %[[EXP2_RESULT:[A-Za-z0-9_]+]] = math.exp2 %[[TENSOR_INPUT]] : tensor<2x29x4xf8E4M3FN>