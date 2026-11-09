// RUN: triton-opt --refine-args-block-id %s | FileCheck %s

// Test for RefineArgsBlockId pass
// This pass moves iter_arg update operations to the block that first uses the iter_arg
//
// In the unfixed version (input):
//   - block_id = 8 uses iter_arg %arg5 in arith.addi
//   - block_id = 33 has the update op for iter_arg (index_cast + addi)
//   - scf.yield yields the result from block_id = 33
//
// After the pass (expected):
//   - The update ops should be moved to block_id = 8
//   - block_id = 33 should no longer exist in the for loop body

module {
  func.func @test_refine_args_block_id() {
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // %ext_value is defined outside the for loop, used inside
    %ext_value = arith.constant 100 : i32

    %alloc_0 = memref.alloc() {ssbuffer.block_id = 1 : i32} : memref<32x32xf32>
    %alloc_1 = memref.alloc() {ssbuffer.block_id = 2 : i32} : memref<32x32xf32>
    %tensor_0 = bufferization.to_tensor %alloc_0 restrict writable {ssbuffer.block_id = 3 : i32} : memref<32x32xf32> to tensor<32x32xf32>
    %tensor_1 = bufferization.to_tensor %alloc_1 restrict writable {ssbuffer.block_id = 4 : i32} : memref<32x32xf32> to tensor<32x32xf32>

    // iter_args: %arg2=tensor, %arg3=tensor, %arg4=index, %arg5=index
    %0:4 = scf.for %arg0 = %c0_i32 to %c32_i32 step %c32_i32 iter_args(%arg2 = %tensor_0, %arg3 = %tensor_1, %arg4 = %c0, %arg5 = %c0) -> (tensor<32x32xf32>, tensor<32x32xf32>, index, index) : i32 {
      // block_id = 6: some computation
      %1 = arith.index_cast %arg0 {ssbuffer.block_id = 6 : i32} : i32 to index
      %2 = arith.addi %arg4, %1 {ssbuffer.block_id = 6 : i32} : index

      // block_id = 8: first user of %arg5 (iter_arg index 3)
      %alloc_8 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32x32xf32>
      %3 = arith.addi %arg5, %c1 {ssbuffer.block_id = 8 : i32} : index

      // block_id = 33: update op for iter_arg %arg5 (to be moved)
      %4 = arith.index_cast %ext_value {ssbuffer.block_id = 33 : i32} : i32 to index
      %5 = arith.addi %arg5, %4 {ssbuffer.block_id = 33 : i32} : index

      scf.yield %tensor_0, %tensor_1, %2, %5 : tensor<32x32xf32>, tensor<32x32xf32>, index, index
    } {DataUse, ssbuffer.main_loop = 1 : i32}

    return
  }
}

// CHECK-LABEL: func.func @test_refine_args_block_id
// After the pass:
// CHECK: arith.index_cast {{.*}} {ssbuffer.block_id = 8 : i32}
// CHECK: arith.addi {{.*}} {ssbuffer.block_id = 8 : i32}
// block_id = 33 should be gone from the for loop body
// CHECK-NOT: ssbuffer.block_id = 33
