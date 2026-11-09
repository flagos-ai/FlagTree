// RUN: triton-opt --add_multi_buffer_inner_scope %s 2>&1 | FileCheck %s
// Pass signals fallback via triton_ascend.dynamic_cv_pipeline.rc = 2
// (ERRCODE_IGNORED); the IR is otherwise unchanged.

// CHECK-LABEL: module attributes
// CHECK-SAME: triton_ascend.dynamic_cv_pipeline.rc = 2

// T26: Memref Type Dependency Triggers Fallback
// Test: When memref.alloc with block_id=X produces a memref, and later
//       bufferization.to_tensor with block_id=Y uses that memref (X != Y),
//       the memref is a cross-block dependency and hasMemrefDepValue returns
//       true, causing pass failure.
// Key Check: Pass falls back via the ERRCODE_FAILED attribute.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_t26_memref_dep_fallback() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant 1.0 : f32
    %empty = tensor.empty() : tensor<128xf32>
    scope.scope : () -> () {
      // Producer in block_id = 5
      %prod = linalg.fill {ssbuffer.block_id = 5 : i32} ins(%cst : f32) outs(%empty : tensor<128xf32>) -> tensor<128xf32>
      // Main loop - all operations inside must have proper ssbuffer.block_id
      %loop_result = scf.for %i = %c0_i32 to %c100_i32 step %c1_i32 iter_args(%arg = %prod) -> (tensor<128xf32>) : i32 {
        // memref.alloc in block_id = 9 (producer block)
        %alloc = memref.alloc() {ssbuffer.block_id = 9 : i32} : memref<128xf32>
        // bufferization.to_tensor in block_id = 10 (consumer block, different from 9)
        // This uses %alloc directly, creating a cross-block memref dependency
        %tensor_from_alloc = bufferization.to_tensor %alloc {ssbuffer.block_id = 10 : i32} : memref<128xf32> to tensor<128xf32>
        // Use the tensor in block_id = 10
        %consumed = arith.addf %tensor_from_alloc, %tensor_from_alloc {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
        // Producer continuation in block_id = 5
        %new_prod = arith.addf %consumed, %arg {ssbuffer.block_id = 5 : i32} : tensor<128xf32>
        scf.yield %new_prod : tensor<128xf32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
