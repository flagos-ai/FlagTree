// RUN: triton-opt --clone-ops --allow-unregistered-dialect %s | FileCheck %s

// Unit test for CloneOps pass: an scf.if whose body only contains
// hivm.hir.sync_block_wait/set ops must be skipped in the
// shouldEraseOpForCube Rule-3 execAfter check, so that earlier cloned
// ops in the same block_id are not blocked by it.
//
// Setup: a CUBE main_loop with two block_ids. Block 10 contains:
//   - annotation.mark  (op with no results, hits Rule 3)
//   - scf.if whose body only has sync_block_wait and sync_block_set
// Block 11 has a sync_block_set (sync op, already skipped by line 232
// in the existing execAfter lambda).
//
// After --clone-ops, block 11 contains (in source order):
//   [cloned_annotation_mark, cloned_ifOp, sync_block_set]
//
// Cleanup processes cloned ops bottom-to-top. For each cloned op,
// shouldEraseOpForCube rebuilds the memGraph and walks the op's
// execAfter. Without the isIfOpWithOnlySyncOps short-circuit, the
// cloned_ifOp would be counted as a same-block_id blocker for the
// cloned_annotation_mark, preventing its erasure. With the fix the
// cloned_ifOp is treated like a sync op (skipped), so the cloned
// annotation_mark is also erased. The cloned_ifOp itself is erased
// in turn because its execAfter contains only the sync_block_set
// (already skipped by line 232).
//
// Expected output: only the original (block_id=10) ops survive in the
// forOp body. Both the cloned annotation.mark and the cloned scf.if
// at block_id=11 are erased.

// CHECK-LABEL: func.func @test_if_with_only_sync_ops_clone_cleanup
// The original scf.if at block 10 is preserved.
// CHECK:         scf.if
// CHECK:           hivm.hir.sync_block_wait
// CHECK:           hivm.hir.sync_block_set
// The cloned annotation.mark at block 11 must be erased (the fix is
// what allows the cloned_ifOp to be skipped in its execAfter check).
// CHECK-NOT:      annotation.mark
// The cloned scf.if at block 11 must also be erased.
// CHECK-NOT:      scf.if
// CHECK:         return

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_if_with_only_sync_ops_clone_cleanup(%arg0: i1, %arg1: memref<4x2x16x8xf32, #hivm.address_space<cbuf>>) attributes {global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10_i32 = arith.constant 10 : i32
    scope.scope : () -> () {
      %alloc = memref.alloc() {ssbuffer.block_id = 10 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
      scf.for %arg2 = %c0_i32 to %c10_i32 step %c1_i32 : i32 {
        // Block 10: annotation.mark (op with no results) + scf.if(only sync ops)
        annotation.mark %alloc {effects = ["write", "read"], ssbuffer.block_id = 10 : i32} : memref<4x2x16x8xf32, #hivm.address_space<cbuf>>
        scf.if %arg0 {
          hivm.hir.sync_block_wait {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
          hivm.hir.sync_block_set {ssbuffer.block_id = 10 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
        } {ssbuffer.block_id = 10 : i32}
        // Block 11: triggers cloning of block 10's ops into block 11
        hivm.hir.sync_block_set {ssbuffer.block_id = 11 : i32, ssbuffer.transfer_id = 0 : i32}[<CUBE>, <PIPE_M>, <PIPE_MTE3>] flag = 2
      } {ssbuffer.block_id = 10 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<CUBE>}
    return
  }
}
