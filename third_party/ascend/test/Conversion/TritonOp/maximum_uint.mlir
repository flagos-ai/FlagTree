// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-a5=False force_simt_template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-a5=False' --split-input-file %s | FileCheck %s

module {
		tt.func public @fn_npu_(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}, %arg3: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<39> : tensor<1x22x1xi32>
		%c39_i32 = arith.constant 39 : i32
		%c22_i32 = arith.constant 22 : i32
		%0 = tt.get_program_id x : i32
		%1 = tt.get_program_id y : i32
		%2 = arith.muli %1, %c22_i32 : i32
		%3 = tt.get_program_id z : i32
		%4 = arith.muli %3, %c39_i32 : i32
		%5 = tt.make_range {end = 22 : i32, start = 0 : i32} : tensor<22xi32>
		%6 = tt.splat %2 : i32 -> tensor<22xi32>
		%7 = arith.addi %5, %6 : tensor<22xi32>
		%8 = tt.make_range {end = 39 : i32, start = 0 : i32} : tensor<39xi32>
		%9 = tt.splat %4 : i32 -> tensor<39xi32>
		%10 = arith.addi %8, %9 : tensor<39xi32>
		%11 = arith.muli %0, %c22_i32 : i32
		%12 = arith.muli %11, %c39_i32 : i32
		%13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<22xi32> -> tensor<1x22xi32>
		%14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x22xi32> -> tensor<1x22x1xi32>
		%15 = arith.muli %14, %cst : tensor<1x22x1xi32>
		%16 = tt.splat %12 : i32 -> tensor<1x22x1xi32>
		%17 = arith.addi %16, %15 : tensor<1x22x1xi32>
		%18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<39xi32> -> tensor<1x39xi32>
		%19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x39xi32> -> tensor<1x1x39xi32>
		%20 = tt.broadcast %17 : tensor<1x22x1xi32> -> tensor<1x22x39xi32>
		%21 = tt.broadcast %19 : tensor<1x1x39xi32> -> tensor<1x22x39xi32>
		%22 = arith.addi %20, %21 : tensor<1x22x39xi32>
		%23 = tt.splat %arg1 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%24 = tt.addptr %23, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		%25 = tt.load %24 : tensor<1x22x39x!tt.ptr>
		%26 = tt.splat %arg2 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%27 = tt.addptr %26, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		%28 = tt.load %27 : tensor<1x22x39x!tt.ptr>
		%29 = arith.maxui %25, %28 : tensor<1x22x39xi8>
		%30 = tt.splat %arg0 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%31 = tt.addptr %30, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		tt.store %31, %29 : tensor<1x22x39x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[VAL_0:[A-Za-z0-9_]+]] = bufferization.to_tensor %alloc restrict writable : memref<1x22x39xi8>
// CHECK: %[[VAL_1:[A-Za-z0-9_]+]] = bufferization.to_tensor %alloc_1 restrict writable : memref<1x22x39xi8>
// CHECK: %[[VAL_2:[A-Za-z0-9_]+]] = arith.maxui %[[VAL_0]], %[[VAL_1]] : tensor<1x22x39xi8>

// -----

module {
		tt.func public @fn_npu_(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}, %arg3: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<39> : tensor<1x22x1xi32>
		%c39_i32 = arith.constant 39 : i32
		%c22_i32 = arith.constant 22 : i32
		%0 = tt.get_program_id x : i32
		%1 = tt.get_program_id y : i32
		%2 = arith.muli %1, %c22_i32 : i32
		%3 = tt.get_program_id z : i32
		%4 = arith.muli %3, %c39_i32 : i32
		%5 = tt.make_range {end = 22 : i32, start = 0 : i32} : tensor<22xi32>
		%6 = tt.splat %2 : i32 -> tensor<22xi32>
		%7 = arith.addi %5, %6 : tensor<22xi32>
		%8 = tt.make_range {end = 39 : i32, start = 0 : i32} : tensor<39xi32>
		%9 = tt.splat %4 : i32 -> tensor<39xi32>
		%10 = arith.addi %8, %9 : tensor<39xi32>
		%11 = arith.muli %0, %c22_i32 : i32
		%12 = arith.muli %11, %c39_i32 : i32
		%13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<22xi32> -> tensor<1x22xi32>
		%14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x22xi32> -> tensor<1x22x1xi32>
		%15 = arith.muli %14, %cst : tensor<1x22x1xi32>
		%16 = tt.splat %12 : i32 -> tensor<1x22x1xi32>
		%17 = arith.addi %16, %15 : tensor<1x22x1xi32>
		%18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<39xi32> -> tensor<1x39xi32>
		%19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x39xi32> -> tensor<1x1x39xi32>
		%20 = tt.broadcast %17 : tensor<1x22x1xi32> -> tensor<1x22x39xi32>
		%21 = tt.broadcast %19 : tensor<1x1x39xi32> -> tensor<1x22x39xi32>
		%22 = arith.addi %20, %21 : tensor<1x22x39xi32>
		%23 = tt.splat %arg1 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%24 = tt.addptr %23, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		%25 = tt.load %24 : tensor<1x22x39x!tt.ptr>
		%26 = tt.splat %arg2 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%27 = tt.addptr %26, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		%28 = tt.load %27 : tensor<1x22x39x!tt.ptr>
		%29 = arith.maxui %25, %28 : tensor<1x22x39xi16>
		%30 = tt.splat %arg0 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%31 = tt.addptr %30, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		tt.store %31, %29 : tensor<1x22x39x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[VAL_0:[A-Za-z0-9_]+]] = bufferization.to_tensor %alloc restrict writable : memref<1x22x39xi16>
// CHECK: %[[VAL_1:[A-Za-z0-9_]+]] = bufferization.to_tensor %alloc_1 restrict writable : memref<1x22x39xi16>
// CHECK: %[[VAL_2:[A-Za-z0-9_]+]] = arith.maxui %[[VAL_0]], %[[VAL_1]] : tensor<1x22x39xi16>

// -----

module {
		tt.func public @fn_npu_(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}, %arg3: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<39> : tensor<1x22x1xi32>
		%c39_i32 = arith.constant 39 : i32
		%c22_i32 = arith.constant 22 : i32
		%0 = tt.get_program_id x : i32
		%1 = tt.get_program_id y : i32
		%2 = arith.muli %1, %c22_i32 : i32
		%3 = tt.get_program_id z : i32
		%4 = arith.muli %3, %c39_i32 : i32
		%5 = tt.make_range {end = 22 : i32, start = 0 : i32} : tensor<22xi32>
		%6 = tt.splat %2 : i32 -> tensor<22xi32>
		%7 = arith.addi %5, %6 : tensor<22xi32>
		%8 = tt.make_range {end = 39 : i32, start = 0 : i32} : tensor<39xi32>
		%9 = tt.splat %4 : i32 -> tensor<39xi32>
		%10 = arith.addi %8, %9 : tensor<39xi32>
		%11 = arith.muli %0, %c22_i32 : i32
		%12 = arith.muli %11, %c39_i32 : i32
		%13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<22xi32> -> tensor<1x22xi32>
		%14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x22xi32> -> tensor<1x22x1xi32>
		%15 = arith.muli %14, %cst : tensor<1x22x1xi32>
		%16 = tt.splat %12 : i32 -> tensor<1x22x1xi32>
		%17 = arith.addi %16, %15 : tensor<1x22x1xi32>
		%18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<39xi32> -> tensor<1x39xi32>
		%19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x39xi32> -> tensor<1x1x39xi32>
		%20 = tt.broadcast %17 : tensor<1x22x1xi32> -> tensor<1x22x39xi32>
		%21 = tt.broadcast %19 : tensor<1x1x39xi32> -> tensor<1x22x39xi32>
		%22 = arith.addi %20, %21 : tensor<1x22x39xi32>
		%23 = tt.splat %arg1 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%24 = tt.addptr %23, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		%25 = tt.load %24 : tensor<1x22x39x!tt.ptr>
		%26 = tt.splat %arg2 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%27 = tt.addptr %26, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		%28 = tt.load %27 : tensor<1x22x39x!tt.ptr>
		%29 = arith.maxui %25, %28 : tensor<1x22x39xi32>
		%30 = tt.splat %arg0 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%31 = tt.addptr %30, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		tt.store %31, %29 : tensor<1x22x39x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[VAL_0:[A-Za-z0-9_]+]] = bufferization.to_tensor %alloc restrict writable : memref<1x22x39xi32>
// CHECK: %[[VAL_1:[A-Za-z0-9_]+]] = bufferization.to_tensor %alloc_1 restrict writable : memref<1x22x39xi32>
// CHECK: %[[VAL_2:[A-Za-z0-9_]+]] = arith.maxui %[[VAL_0]], %[[VAL_1]] : tensor<1x22x39xi32>

// -----

module {
		tt.func public @fn_npu_(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}, %arg3: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%cst = arith.constant dense<39> : tensor<1x22x1xi32>
		%c39_i32 = arith.constant 39 : i32
		%c22_i32 = arith.constant 22 : i32
		%0 = tt.get_program_id x : i32
		%1 = tt.get_program_id y : i32
		%2 = arith.muli %1, %c22_i32 : i32
		%3 = tt.get_program_id z : i32
		%4 = arith.muli %3, %c39_i32 : i32
		%5 = tt.make_range {end = 22 : i32, start = 0 : i32} : tensor<22xi32>
		%6 = tt.splat %2 : i32 -> tensor<22xi32>
		%7 = arith.addi %5, %6 : tensor<22xi32>
		%8 = tt.make_range {end = 39 : i32, start = 0 : i32} : tensor<39xi32>
		%9 = tt.splat %4 : i32 -> tensor<39xi32>
		%10 = arith.addi %8, %9 : tensor<39xi32>
		%11 = arith.muli %0, %c22_i32 : i32
		%12 = arith.muli %11, %c39_i32 : i32
		%13 = tt.expand_dims %7 {axis = 0 : i32} : tensor<22xi32> -> tensor<1x22xi32>
		%14 = tt.expand_dims %13 {axis = 2 : i32} : tensor<1x22xi32> -> tensor<1x22x1xi32>
		%15 = arith.muli %14, %cst : tensor<1x22x1xi32>
		%16 = tt.splat %12 : i32 -> tensor<1x22x1xi32>
		%17 = arith.addi %16, %15 : tensor<1x22x1xi32>
		%18 = tt.expand_dims %10 {axis = 0 : i32} : tensor<39xi32> -> tensor<1x39xi32>
		%19 = tt.expand_dims %18 {axis = 1 : i32} : tensor<1x39xi32> -> tensor<1x1x39xi32>
		%20 = tt.broadcast %17 : tensor<1x22x1xi32> -> tensor<1x22x39xi32>
		%21 = tt.broadcast %19 : tensor<1x1x39xi32> -> tensor<1x22x39xi32>
		%22 = arith.addi %20, %21 : tensor<1x22x39xi32>
		%23 = tt.splat %arg1 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%24 = tt.addptr %23, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		%25 = tt.load %24 : tensor<1x22x39x!tt.ptr>
		%26 = tt.splat %arg2 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%27 = tt.addptr %26, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		%28 = tt.load %27 : tensor<1x22x39x!tt.ptr>
		%29 = arith.maxui %25, %28 : tensor<1x22x39xi64>
		%30 = tt.splat %arg0 : !tt.ptr -> tensor<1x22x39x!tt.ptr>
		%31 = tt.addptr %30, %22 : tensor<1x22x39x!tt.ptr>, tensor<1x22x39xi32>
		tt.store %31, %29 : tensor<1x22x39x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[VAL_0:[A-Za-z0-9_]+]] = bufferization.to_tensor %alloc restrict writable : memref<1x22x39xi64>
// CHECK: %[[VAL_1:[A-Za-z0-9_]+]] = bufferization.to_tensor %alloc_1 restrict writable : memref<1x22x39xi64>
// CHECK: %[[VAL_2:[A-Za-z0-9_]+]] = arith.maxui %[[VAL_0]], %[[VAL_1]] : tensor<1x22x39xi64>

