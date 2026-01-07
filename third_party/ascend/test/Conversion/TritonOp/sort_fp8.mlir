// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-a5=False force_simt_template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-a5=False' --split-input-file %s | FileCheck %s

module {
		tt.func public @sort_kernel_3d(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%c116_i32 = arith.constant 116 : i32
		%c4_i32 = arith.constant 4 : i32
		%c29_i32 = arith.constant 29 : i32
		%0 = tt.get_program_id x : i32
		%1 = arith.remsi %0, %c29_i32 : i32
		%2 = arith.divsi %0, %c29_i32 : i32
		%3 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
		%4 = arith.muli %1, %c4_i32 : i32
		%5 = arith.muli %2, %c116_i32 : i32
		%6 = tt.splat %4 : i32 -> tensor<4xi32>
		%7 = arith.addi %3, %6 : tensor<4xi32>
		%8 = tt.splat %5 : i32 -> tensor<4xi32>
		%9 = arith.addi %7, %8 : tensor<4xi32>
		%10 = tt.splat %arg0 : !tt.ptr -> tensor<4x!tt.ptr>
		%11 = tt.addptr %10, %9 : tensor<4x!tt.ptr>, tensor<4xi32>
		%12 = tt.load %11 : tensor<4x!tt.ptr>
		%13 = tt.sort %12, 0, false : tensor<4xf8E4M3FN> -> tensor<4xf8E4M3FN>
		%14 = tt.splat %arg1 : !tt.ptr -> tensor<4x!tt.ptr>
		%15 = tt.addptr %14, %9 : tensor<4x!tt.ptr>, tensor<4xi32>
		tt.store %15, %13 : tensor<4x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[ALLOC_INPUT:[A-Za-z0-9_]+]] = memref.alloc() : memref<4xf8E4M3FN>
// CHECK: %[[TENSOR_INPUT:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC_INPUT]] restrict writable : memref<4xf8E4M3FN>

// CHECK: %[[SORT_RESULT:[A-Za-z0-9_]+]] = call @triton_sort(%[[TENSOR_INPUT]], %c0_i64, %false) : (tensor<4xf8E4M3FN>, i64, i1) -> tensor<4xf8E4M3FN>

// -----

module {
		tt.func public @sort_kernel_3d(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%c116_i32 = arith.constant 116 : i32
		%c4_i32 = arith.constant 4 : i32
		%c29_i32 = arith.constant 29 : i32
		%0 = tt.get_program_id x : i32
		%1 = arith.remsi %0, %c29_i32 : i32
		%2 = arith.divsi %0, %c29_i32 : i32
		%3 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
		%4 = arith.muli %1, %c4_i32 : i32
		%5 = arith.muli %2, %c116_i32 : i32
		%6 = tt.splat %4 : i32 -> tensor<4xi32>
		%7 = arith.addi %3, %6 : tensor<4xi32>
		%8 = tt.splat %5 : i32 -> tensor<4xi32>
		%9 = arith.addi %7, %8 : tensor<4xi32>
		%10 = tt.splat %arg0 : !tt.ptr -> tensor<4x!tt.ptr>
		%11 = tt.addptr %10, %9 : tensor<4x!tt.ptr>, tensor<4xi32>
		%12 = tt.load %11 : tensor<4x!tt.ptr>
		%13 = tt.sort %12, 0, false : tensor<4xf8E5M2> -> tensor<4xf8E5M2>
		%14 = tt.splat %arg1 : !tt.ptr -> tensor<4x!tt.ptr>
		%15 = tt.addptr %14, %9 : tensor<4x!tt.ptr>, tensor<4xi32>
		tt.store %15, %13 : tensor<4x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[ALLOC_INPUT:[A-Za-z0-9_]+]] = memref.alloc() : memref<4xf8E5M2>
// CHECK: %[[TENSOR_INPUT:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC_INPUT]] restrict writable : memref<4xf8E5M2>

// CHECK: %[[SORT_RESULT:[A-Za-z0-9_]+]] = call @triton_sort(%[[TENSOR_INPUT]], %c0_i64, %false) : (tensor<4xf8E5M2>, i64, i1) -> tensor<4xf8E5M2>