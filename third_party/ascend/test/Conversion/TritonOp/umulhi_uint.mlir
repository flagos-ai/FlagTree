// RUN: triton-adapter-opt --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-a5=False force_simt_template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-a5=False' %s | FileCheck %s

module {
		tt.func public @umulhi_kernel(%arg0: !tt.ptr {tt.divisibility = 16 : i32}, %arg1: !tt.ptr {tt.divisibility = 16 : i32}, %arg2: !tt.ptr {tt.divisibility = 16 : i32}) attributes {noinline = false} {
		%c0_i32 = arith.constant 0 : i32
		%0 = tt.addptr %arg0, %c0_i32 : !tt.ptr, i32
		%1 = tt.splat %0 : !tt.ptr -> tensor<1x!tt.ptr>
		%2 = tt.load %1 : tensor<1x!tt.ptr>
		%3 = tt.addptr %arg1, %c0_i32 : !tt.ptr, i32
		%4 = tt.splat %3 : !tt.ptr -> tensor<1x!tt.ptr>
		%5 = tt.load %4 : tensor<1x!tt.ptr>
		%6 = tt.mulhiui %2, %5 : tensor<1xi64>
		%7 = tt.addptr %arg2, %c0_i32 : !tt.ptr, i32
		%8 = tt.splat %7 : !tt.ptr -> tensor<1x!tt.ptr>
		tt.store %8, %6 : tensor<1x!tt.ptr>
		tt.return
	}
}

// CHECK: %[[CAST1:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
// CHECK: %[[ALLOC1:[A-Za-z0-9_]+]] = memref.alloc() : memref<1xi64>
// CHECK: memref.copy %[[CAST1]], %[[ALLOC1]] : memref<1xi64, strided<[1]>> to memref<1xi64>
// CHECK: %[[TENSOR1:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC1]] restrict writable : memref<1xi64>

// CHECK: %[[CAST2:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg3 to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
// CHECK: %[[ALLOC2:[A-Za-z0-9_]+]] = memref.alloc() : memref<1xi64>
// CHECK: memref.copy %[[CAST2]], %[[ALLOC2]] : memref<1xi64, strided<[1]>> to memref<1xi64>
// CHECK: %[[TENSOR2:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC2]] restrict writable : memref<1xi64>

// CHECK: %[[LOW:[A-Za-z0-9_]+]], %[[HIGH:[A-Za-z0-9_]+]] = arith.mulsi_extended %[[TENSOR1]], %[[TENSOR2]] : tensor<1xi64>

// CHECK: %[[CAST3:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg4 to offset: [0], sizes: [1], strides: [1] : memref<?xi64> to memref<1xi64, strided<[1]>>
// CHECK: bufferization.materialize_in_destination %[[HIGH]] in writable %[[CAST3]] : (tensor<1xi64>, memref<1xi64, strided<[1]>>) -> ()