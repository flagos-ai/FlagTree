// RUN: triton-adapter-opt --triton-linearize '--discrete-mask-access-conversion=compile-on-a5=True force_simt_template=False' --triton-to-annotation '--triton-to-unstructure=compile-on-a5=True force_simt_template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-a5=True' --split-input-file %s | FileCheck %s

module {
		tt.func public @fn_npu_(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}, %arg3: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<4> : tensor<2x1x1xi32>
		%cst_0 = arith.constant dense<29> : tensor<2x1x1xi32>
		%c4_i32 = arith.constant 4 : i32
		%c2_i32 = arith.constant 2 : i32
		%0 = tt.get_program_id x : i32
		%1 = arith.muli %0, %c2_i32 : i32
		%2 = tt.get_program_id y : i32
		%3 = tt.get_program_id z : i32
		%4 = arith.muli %3, %c4_i32 : i32
		%5 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
		%6 = tt.splat %1 : i32 -> tensor<2xi32>
		%7 = arith.addi %5, %6 : tensor<2xi32>
		%8 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
		%9 = tt.splat %4 : i32 -> tensor<4xi32>
		%10 = arith.addi %8, %9 : tensor<4xi32>
		%11 = tt.expand_dims %7 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
		%12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
		%13 = arith.muli %12, %cst_0 : tensor<2x1x1xi32>
		%14 = arith.muli %13, %cst : tensor<2x1x1xi32>
		%15 = arith.muli %2, %c4_i32 : i32
		%16 = tt.splat %15 : i32 -> tensor<2x1x1xi32>
		%17 = arith.addi %14, %16 : tensor<2x1x1xi32>
		%18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
		%19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
		%20 = tt.broadcast %17 : tensor<2x1x1xi32> -> tensor<2x1x4xi32>
		%21 = tt.broadcast %19 : tensor<1x1x4xi32> -> tensor<2x1x4xi32>
		%22 = arith.addi %20, %21 : tensor<2x1x4xi32>
		%23 = tt.splat %arg1 : !tt.ptr -> tensor<2x1x4x!tt.ptr>
		%24 = tt.addptr %23, %22 : tensor<2x1x4x!tt.ptr>, tensor<2x1x4xi32>
		%25 = tt.load %24 : tensor<2x1x4x!tt.ptr>
		%26 = math.log2 %25 : tensor<2x1x4xf8E4M3FN>
		%27 = tt.splat %arg0 : !tt.ptr -> tensor<2x1x4x!tt.ptr>
		%28 = tt.addptr %27, %22 : tensor<2x1x4x!tt.ptr>, tensor<2x1x4xi32>
		tt.store %28, %26 : tensor<2x1x4x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[ALLOC_INPUT:[A-Za-z0-9_]+]] = memref.alloc() : memref<2x1x4xf8E4M3FN>
// CHECK: %[[TENSOR_INPUT:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC_INPUT]] restrict writable : memref<2x1x4xf8E4M3FN>
// CHECK: %[[LOG2_RESULT:[A-Za-z0-9_]+]] = math.log2 %[[TENSOR_INPUT]] : tensor<2x1x4xf8E4M3FN>

// -----

module {
		tt.func public @fn_npu_(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}, %arg3: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<4> : tensor<2x1x1xi32>
		%cst_0 = arith.constant dense<29> : tensor<2x1x1xi32>
		%c4_i32 = arith.constant 4 : i32
		%c2_i32 = arith.constant 2 : i32
		%0 = tt.get_program_id x : i32
		%1 = arith.muli %0, %c2_i32 : i32
		%2 = tt.get_program_id y : i32
		%3 = tt.get_program_id z : i32
		%4 = arith.muli %3, %c4_i32 : i32
		%5 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
		%6 = tt.splat %1 : i32 -> tensor<2xi32>
		%7 = arith.addi %5, %6 : tensor<2xi32>
		%8 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
		%9 = tt.splat %4 : i32 -> tensor<4xi32>
		%10 = arith.addi %8, %9 : tensor<4xi32>
		%11 = tt.expand_dims %7 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
		%12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
		%13 = arith.muli %12, %cst_0 : tensor<2x1x1xi32>
		%14 = arith.muli %13, %cst : tensor<2x1x1xi32>
		%15 = arith.muli %2, %c4_i32 : i32
		%16 = tt.splat %15 : i32 -> tensor<2x1x1xi32>
		%17 = arith.addi %14, %16 : tensor<2x1x1xi32>
		%18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
		%19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x4xi32> -> tensor<1x1x4xi32>
		%20 = tt.broadcast %17 : tensor<2x1x1xi32> -> tensor<2x1x4xi32>
		%21 = tt.broadcast %19 : tensor<1x1x4xi32> -> tensor<2x1x4xi32>
		%22 = arith.addi %20, %21 : tensor<2x1x4xi32>
		%23 = tt.splat %arg1 : !tt.ptr -> tensor<2x1x4x!tt.ptr>
		%24 = tt.addptr %23, %22 : tensor<2x1x4x!tt.ptr>, tensor<2x1x4xi32>
		%25 = tt.load %24 : tensor<2x1x4x!tt.ptr>
		%26 = math.log2 %25 : tensor<2x1x4xf8E5M2>
		%27 = tt.splat %arg0 : !tt.ptr -> tensor<2x1x4x!tt.ptr>
		%28 = tt.addptr %27, %22 : tensor<2x1x4x!tt.ptr>, tensor<2x1x4xi32>
		tt.store %28, %26 : tensor<2x1x4x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[ALLOC_INPUT:[A-Za-z0-9_]+]] = memref.alloc() : memref<2x1x4xf8E5M2>
// CHECK: %[[TENSOR_INPUT:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC_INPUT]] restrict writable : memref<2x1x4xf8E5M2>
// CHECK: %[[LOG2_RESULT:[A-Za-z0-9_]+]] = math.log2 %[[TENSOR_INPUT]] : tensor<2x1x4xf8E5M2>