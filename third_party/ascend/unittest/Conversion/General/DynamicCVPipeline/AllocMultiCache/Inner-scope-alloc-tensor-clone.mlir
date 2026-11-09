// RUN: triton-opt --add_multi_buffer_inner_scope %s | FileCheck %s

// T-alloc-tensor-clone: bufferization.alloc_tensor() (no payload) is cloned
// into each cross-block consumer's block instead of being multi-buffered.
//
// Case A (test_alloc_tensor_single_consumer):
//   %alloc in block_id = 9 (producer block), consumed by linalg.fill in
//   block_id = 10. After the pass:
//     - Original %alloc stays at block_id = 9 (orphaned, DCE will clean it).
//     - A cloned alloc_tensor appears at block_id = 10 (the consumer block).
//     - linalg.fill's outs is rewired to the clone, NOT the original.
//     - NO multi-buffer machinery (memref.alloc + hivm.copy + scf.if +
//       to_tensor) is generated for this dep.
//
// Case B (test_alloc_tensor_multi_consumer):
//   %alloc in block_id = 9, consumed by two ops in two different consumer
//   blocks (block_id = 10 and block_id = 11). After the pass, the alloc is
//   cloned ONCE PER consumer block — each consumer gets a fresh local alloc
//   with the corresponding block_id tag.

// CHECK-LABEL: func.func @test_alloc_tensor_single_consumer
// Original alloc stays in producer block (orphaned, will be DCE'd).
// CHECK: %[[ORIG:.*]] = bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32} : tensor<f32>
// Cloned alloc appears in consumer block, tagged with consumer's block_id.
// CHECK: %[[CLONE:.*]] = bufferization.alloc_tensor() {ssbuffer.block_id = 10 : i32} : tensor<f32>
// linalg.fill's outs is rewired to the clone, NOT the original.
// CHECK: linalg.fill {ssbuffer.block_id = 10 : i32} ins(%{{.+}} : f32) outs(%[[CLONE]] : tensor<f32>) -> tensor<f32>
// alloc_tensor must NOT go through the multi-buffer path: no UB memref.alloc
// or hivm.hir.copy chain for tensor<f32> in this scope.
// CHECK-NOT: memref.alloc() : memref<f32, #hivm.address_space<ub>>
// CHECK-NOT: hivm.hir.copy{{.*}}: tensor<f32>

// CHECK-LABEL: func.func @test_alloc_tensor_multi_consumer
// Original alloc stays in producer block (orphaned).
// CHECK: %[[B_ORIG:.*]] = bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32} : tensor<f32>
// Clone for block_id = 10 consumer.
// CHECK: %[[B_CLONE_10:.*]] = bufferization.alloc_tensor() {ssbuffer.block_id = 10 : i32} : tensor<f32>
// block_id = 10 fill uses the block 10 clone, NOT the original.
// CHECK: linalg.fill {ssbuffer.block_id = 10 : i32} ins(%{{.+}} : f32) outs(%[[B_CLONE_10]] : tensor<f32>) -> tensor<f32>
// Clone for block_id = 11 consumer.
// CHECK: %[[B_CLONE_11:.*]] = bufferization.alloc_tensor() {ssbuffer.block_id = 11 : i32} : tensor<f32>
// block_id = 11 fill uses the block 11 clone, NOT the original or the block 10
// clone.
// CHECK: linalg.fill {ssbuffer.block_id = 11 : i32} ins(%{{.+}} : f32) outs(%[[B_CLONE_11]] : tensor<f32>) -> tensor<f32>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Case A: single cross-block consumer
  func.func @test_alloc_tensor_single_consumer() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
        // alloc_tensor in producer block (block_id = 9).
        %alloc = bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32} : tensor<f32>
        // Single cross-block consumer (block_id = 10). Uses %alloc as outs.
        %fill = linalg.fill {ssbuffer.block_id = 10 : i32} ins(%cst : f32) outs(%alloc : tensor<f32>) -> tensor<f32>
        %used = arith.addf %fill, %fill {ssbuffer.block_id = 10 : i32} : tensor<f32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // Case B: two cross-block consumers in different blocks
  func.func @test_alloc_tensor_multi_consumer() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
        %alloc = bufferization.alloc_tensor() {ssbuffer.block_id = 9 : i32} : tensor<f32>
        // Two cross-block consumers, different block_ids: 10 and 11.
        %fill_a = linalg.fill {ssbuffer.block_id = 10 : i32} ins(%cst : f32) outs(%alloc : tensor<f32>) -> tensor<f32>
        %fill_b = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%cst : f32) outs(%alloc : tensor<f32>) -> tensor<f32>
        %used_a = arith.addf %fill_a, %fill_a {ssbuffer.block_id = 10 : i32} : tensor<f32>
        %used_b = arith.addf %fill_b, %fill_b {ssbuffer.block_id = 11 : i32} : tensor<f32>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
