// RUN: triton-opt --relocate-memref-decl %s | FileCheck %s

// ============================================================================
// RelocateMemrefDeclPass: scan every op for memref-typed operands; when the
// operand's producer crosses a sync op relative to the consumer, sink the
// producer (and its forward/backward closure) into the consumer's block and
// re-partition it to the consumer's block_id group.
//
// Covered scenarios:
//   1. @relocate_across_sync     - a memref.alloc declared before a gpu.barrier
//                                  whose only consumer (memref.load) is after
//                                  the barrier: the alloc is moved to just
//                                  before the load and re-partitioned to the
//                                  consumer's block (1 -> 2).
//   2. @same_side_no_relocation  - producer and consumer both on the same side
//                                  of the barrier (no sync in between): nothing
//                                  is moved, block ids unchanged.
// ============================================================================

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Producer (block 1) before the barrier, consumer (block 2) after it. The
  // alloc must be pulled across the barrier to just before the load and given
  // the consumer's block id.
  // CHECK-LABEL: func.func @relocate_across_sync
  func.func @relocate_across_sync() {
    %c0 = arith.constant 0 : index
    // CHECK: arith.constant 0 : index
    // CHECK-NOT: memref.alloc
    %alloc = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : memref<32x32xf32>
    // CHECK: gpu.barrier {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.external_sync = 1 : i32}
    gpu.barrier {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.external_sync = 1 : i32}
    // CHECK: memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<32x32xf32>
    // CHECK: memref.load %{{.*}} {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<32x32xf32>
    %ld = memref.load %alloc[%c0, %c0] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "VECTOR"} : memref<32x32xf32>
    return
  }

  // Producer and consumer share the same side of the barrier, so there is no
  // sync between them; the pass must leave everything untouched.
  // CHECK-LABEL: func.func @same_side_no_relocation
  func.func @same_side_no_relocation() {
    %c0 = arith.constant 0 : index
    // CHECK: arith.constant 0 : index
    // CHECK: memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : memref<16x16xf32>
    %alloc = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : memref<16x16xf32>
    // CHECK: memref.load %{{.*}} {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : memref<16x16xf32>
    %ld = memref.load %alloc[%c0, %c0] {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : memref<16x16xf32>
    // CHECK: gpu.barrier {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.external_sync = 1 : i32}
    gpu.barrier {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "CUBE", ssbuffer.external_sync = 1 : i32}
    return
  }
}
