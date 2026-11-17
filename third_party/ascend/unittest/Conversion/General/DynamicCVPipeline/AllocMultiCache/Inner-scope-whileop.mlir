// RUN: triton-opt --add_multi_buffer_inner_scope %s | FileCheck %s

// T-while-A: whileOp as main_loop, INTRA bufNum == 1 (single-buffer scope).
//   Verifies:
//     - whileOp carrying ssbuffer.main_loop on its terminator IS recognized
//       (legacy shape supported by hasMainLoopAttr).
//     - setupWhileIterArgCounter is SKIPPED (bufNum==1 → no dead iter_arg).
//     - The memref.alloc + bufferization.to_tensor dep is CLONED into the
//       consumer block (NOT multi-buffered) — the alloc has no payload on
//       entry, so multi-buffering would be a read-before-write copy.
//     - NO UB memref.alloc or hivm.hir.copy is emitted for this dep.
//     - The original whileOp is preserved verbatim (no extra i32 iter_arg).
//   We pin the buffer count to 1 via the module-level
//   `ssbuffer.intra_buf_count` attribute so the default of 2 doesn't
//   trigger the counter-setup branch.

// CHECK-LABEL: func.func @test_while_mainloop_bufnum_one
// Original whileOp do-region bb0 has only 2 block-args (no i32 counter).
// CHECK:       ^bb0(%{{.*}}: tensor<128xf32>, %{{.*}}: i32):
// Original alloc + to_tensor stay in producer block (orphaned, will be DCE'd).
// CHECK-DAG:   memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xf32>
// CHECK-DAG:   bufferization.to_tensor {{.*}}{ssbuffer.block_id = 7 : i32} : memref<128xf32> to tensor<128xf32>
// Cloned alloc + to_tensor appear in consumer block (block_id = 10).
// CHECK-DAG:   memref.alloc() {ssbuffer.block_id = 10 : i32} : memref<128xf32>
// CHECK-DAG:   bufferization.to_tensor {{.*}}{ssbuffer.block_id = 10 : i32} : memref<128xf32> to tensor<128xf32>
// Consumer uses the cloned to_tensor at block_id = 10.
// CHECK:       arith.addf {{.*}}{ssbuffer.block_id = 10 : i32} : tensor<128xf32>
// NO UB alloc, NO hivm.copy chain — dep was cloned, not multi-buffered.
// CHECK-NOT:   memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
// CHECK-NOT:   hivm.hir.copy{{.*}}: tensor<128xf32>
// whileOp carries main_loop but NOT iterCounter (bufNum == 1).
// CHECK:       } {{.*}}ssbuffer.main_loop = 1 : i64
// CHECK-NOT:   } {{.*}}ssbuffer.iterCounter,

// T-while-B: whileOp as main_loop, INTRA bufNum == 2 (multi-buffer scope).
//   Verifies:
//     - setupWhileIterArgCounter IS called (bufNum>1) — the whileOp gains
//       an extra i32 iter_arg (init=0, yielded as counter+1 at end of
//       do-region).
//     - The new do-region bb0 has 3 block-args (counter arg appended).
//     - The memref.alloc + to_tensor dep is CLONED to the consumer block
//       (NOT multi-buffered), so no UB alloc + scf.if dispatch + to_tensor
//       readback chain is emitted.
//     - The counter add-1 (and its constant 1) is relocated to the
//       block_id of the FIRST OP that consumes the counter arg — which is
//       now the cloned alloc at block_id = 10 (the alloc+to_tensor clone
//       becomes the new "first user" since the multi-buffer producer scf.if
//       is gone).

// CHECK-LABEL: func.func @test_while_mainloop_bufnum_two
// WhileOp's do-region bb0 has 3 block-args now (the new i32 counter is last).
// CHECK:       ^bb0(%{{.*}}: tensor<128xf32>, %{{.*}}: i32, %{{.*}}: i32):
// Original alloc + to_tensor stay in producer block (orphaned).
// CHECK-DAG:   memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xf32>
// CHECK-DAG:   bufferization.to_tensor {{.*}}{ssbuffer.block_id = 7 : i32} : memref<128xf32> to tensor<128xf32>
// Cloned alloc + to_tensor appear in consumer block (block_id = 10).
// CHECK-DAG:   memref.alloc() {ssbuffer.block_id = 10 : i32} : memref<128xf32>
// CHECK-DAG:   bufferization.to_tensor {{.*}}{ssbuffer.block_id = 10 : i32} : memref<128xf32> to tensor<128xf32>
// Consumer uses the cloned to_tensor at block_id = 10.
// CHECK:       arith.addf {{.*}}{ssbuffer.block_id = 10 : i32} : tensor<128xf32>
// NO UB alloc, NO scf.if dispatch, NO hivm.copy — dep was cloned, not
// multi-buffered. (With bufNum==2 + non-cloneable dep, multi-buffer would
// have generated the full chain.)
// CHECK-NOT:   memref.alloc() : memref<128xf32, #hivm.address_space<ub>>
// CHECK-NOT:   scf.if
// CHECK-NOT:   hivm.hir.copy{{.*}}: tensor<128xf32>
// arith.addi increment for the multi-buffer counter — relocated to the
// first consumer's block_id (= 10, the cloned alloc), NOT 7.
// CHECK:       %{{.+}} = arith.addi %{{.+}}, %{{.+}} {ssbuffer.block_id = 10 : i32, ssbuffer.iterCounter} : i32
// Counter-aware whileOp carries iterCounter alongside main_loop.
// CHECK:       } {{.*}}ssbuffer.iterCounter, {{.*}}ssbuffer.main_loop = 1 : i64

// T-while-C: whileOp main_loop with scf.if inside the do-region (multi-region
// consumer pattern). The whileOp carries `ssbuffer.main_loop` on itself (NOT
// on the terminator). This exercises `hasMainLoopAttr`'s "op has attr" path,
// in contrast to T-while-A/B which use the terminator-attr legacy shape.
// Verifies the cross-block dep is cloned into BOTH consumer branches.

// CHECK-LABEL: func.func @test_while_mainloop_attr_on_op
// Original alloc + to_tensor stay in producer block (orphaned).
// CHECK:       %[[C_ORIG_ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<64xf16>
// CHECK:       %[[C_ORIG_TT:.*]] = bufferization.to_tensor %[[C_ORIG_ALLOC]] {ssbuffer.block_id = 7 : i32} : memref<64xf16> to tensor<64xf16>
// Cloned alloc + to_tensor appear in BOTH consumer blocks (block_id = 11 in
// the if-branch, block_id = 12 in the else-branch).
// CHECK:       %[[C_CLONE_ALLOC_11:.*]] = memref.alloc() {ssbuffer.block_id = 11 : i32} : memref<64xf16>
// CHECK:       %[[C_CLONE_TT_11:.*]] = bufferization.to_tensor %[[C_CLONE_ALLOC_11]] {ssbuffer.block_id = 11 : i32} : memref<64xf16> to tensor<64xf16>
// CHECK:       arith.addf %[[C_CLONE_TT_11]], %[[C_CLONE_TT_11]] {ssbuffer.block_id = 11 : i32} : tensor<64xf16>
// CHECK:       %[[C_CLONE_ALLOC_12:.*]] = memref.alloc() {ssbuffer.block_id = 12 : i32} : memref<64xf16>
// CHECK:       %[[C_CLONE_TT_12:.*]] = bufferization.to_tensor %[[C_CLONE_ALLOC_12]] {ssbuffer.block_id = 12 : i32} : memref<64xf16> to tensor<64xf16>
// CHECK:       arith.mulf %[[C_CLONE_TT_12]], %[[C_CLONE_TT_12]] {ssbuffer.block_id = 12 : i32} : tensor<64xf16>
// NO UB alloc, NO consumer-side to_tensor readback — dep was cloned.
// CHECK-NOT:   memref.alloc() : memref<64xf16, #hivm.address_space<ub>>
// CHECK-NOT:   hivm.hir.copy{{.*}}: tensor<64xf16>
// Counter add-1 relocated to first consumer's block_id (= 12, the cloned
// alloc's location).
// CHECK:       %{{.+}} = arith.addi %{{.+}}, %{{.+}} {ssbuffer.block_id = 12 : i32, ssbuffer.iterCounter} : i32
// Counter-aware whileOp carries iterCounter alongside main_loop.
// CHECK:       } {{.*}}ssbuffer.iterCounter, {{.*}}ssbuffer.main_loop = 1 : i64

// T-while-D: Regression guard for the getOutermostSsbufferId priority fix.
//   Before the fix, walking up from an op inside a whileOp's do-region would
//   hit the multi-region check (`numRegions >= 2`) BEFORE the kMainLoop check
//   and return the whileOp's own block_id, breaking cross-block dep
//   classification.
//   Setup: whileOp (block_id = 5) wrapping a producer block_id = 8 inside the
//   do-region and a consumer block_id = 12 also inside the do-region.
//   Cross-block judgment must use 8 vs 12 (NOT 5 vs 12). After my fix the dep
//   is cloned to block_id = 12; if the priority bug were present the producer
//   and consumer would both be classified as block_id = 5, the dep would NOT
//   be detected as cross-block, and NO clone would be emitted.

// CHECK-LABEL: func.func @test_while_outermost_id_priority
// Original alloc + to_tensor stay at producer block (block_id = 8).
// CHECK:       %[[D_ORIG_ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32xf32>
// CHECK:       %[[D_ORIG_TT:.*]] = bufferization.to_tensor %[[D_ORIG_ALLOC]] {ssbuffer.block_id = 8 : i32} : memref<32xf32> to tensor<32xf32>
// Cloned alloc + to_tensor appear at consumer block (block_id = 12). If the
// outermost-id priority bug were present, no clone would be emitted (the dep
// would be misclassified as same-block and silently skipped).
// CHECK:       %[[D_CLONE_ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = 12 : i32} : memref<32xf32>
// CHECK:       %[[D_CLONE_TT:.*]] = bufferization.to_tensor %[[D_CLONE_ALLOC]] {ssbuffer.block_id = 12 : i32} : memref<32xf32> to tensor<32xf32>
// CHECK:       arith.addf %[[D_CLONE_TT]], %[[D_CLONE_TT]] {ssbuffer.block_id = 12 : i32} : tensor<32xf32>
// NO UB alloc, NO copy chain.
// CHECK-NOT:   memref.alloc() : memref<32xf32, #hivm.address_space<ub>>
// CHECK-NOT:   hivm.hir.copy{{.*}}: tensor<32xf32>
// Counter add-1 at first consumer's block_id (= 12).
// CHECK:       %{{.+}} = arith.addi %{{.+}}, %{{.+}} {ssbuffer.block_id = 12 : i32, ssbuffer.iterCounter} : i32
// Counter-aware whileOp carries iterCounter alongside main_loop.
// CHECK:       } {{.*}}ssbuffer.iterCounter, {{.*}}ssbuffer.main_loop = 1 : i64

// T-while-E: whileOp main_loop with a tensor::EmptyOp + linalg::FillOp pattern
//   inside the do-region. The empty+fill is cloned into each consumer block.
//   Verifies cloneEmptyFillsInBlocks works on a whileOp main_loop (the body
//   is `getAfterBody()`, not `getBody()`).
//   The cloned fill's ins stays the same (arith.constant, lives outside the
//   main_loop) — no clone of the scalar chain is needed.
//   NOTE: This case uses tensor.empty + linalg.fill (NOT memref.alloc +
//   to_tensor), so the alloc-to-tensor clone logic doesn't apply. The empty+
//   fill clone is exercised here, unchanged.

// CHECK-LABEL: func.func @test_while_clone_empty_fill
// Original fill at block_id = 8 (producer block) is preserved.
// CHECK:       linalg.fill {{.*}}{ssbuffer.block_id = 8 : i32} {{.*}}outs({{.*}} : tensor<32x1xf32>)
// Original fill at block_id = 14 (second producer block) is preserved.
// CHECK:       linalg.fill {{.*}}{ssbuffer.block_id = 14 : i32} {{.*}}outs({{.*}} : tensor<32x1xf32>)
// Cloned fills land in block_id = 12 (consumer block) — there are two of them,
// one per original producer. CHECK matches at least once.
// CHECK:       linalg.fill {{.*}}{ssbuffer.block_id = 12 : i32} {{.*}}outs({{.*}} : tensor<32x1xf32>)
// arith.addi increment for the multi-buffer counter is present (block_id=12), tagged iterCounter.
// CHECK:       %{{.+}} = arith.addi %{{.+}}, %{{.+}} {ssbuffer.block_id = 12 : i32, ssbuffer.iterCounter} : i32
// Counter-aware whileOp carries ssbuffer.iterCounter alongside main_loop.
// CHECK:       } {{.*}}ssbuffer.iterCounter, {{.*}}ssbuffer.main_loop = 1 : i64

// T-while-F (TODO): Negative — two whileOps in the same scope, neither is the
//   "outer" main_loop; one is nested in the other's do-region. The pass must
//   refuse via findNestedMainloop (returns WalkResult::interrupt + fallback).
//   Not implemented in this initial drop — to be added when the negative case
//   is needed for a regression guard.

// T-while-G: Regression guard for relocateWhileCounterOps.
//   The counter add-1 (and its constant 1) used to be appended at the end of
//   the do-region, carrying the whileOp's own block_id (= 99 here, distinct
//   from both producer and consumer). The pass now relocates them to the
//   block_id of the first op that consumes the counter iter_arg — which is
//   now the cloned alloc at block_id = 50 (the alloc+to_tensor clone becomes
//   the new "first user" since multi-buffer producer scf.if is gone).
//   We assert:
//     - The counter add-1 carries block_id = 50 (NOT 99, NOT 5).
//     - The constant 1 used by the addi also carries block_id = 50.
//     - The scf.yield that consumes the add-1's SSA value still references
//       the same %result (SSA is preserved across the move).

// CHECK-LABEL: func.func @test_while_counter_relocation
// Original alloc + to_tensor stay at producer block (block_id = 5, orphaned).
// CHECK:       %[[G_ORIG_ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<16xf32>
// CHECK:       %[[G_ORIG_TT:.*]] = bufferization.to_tensor %[[G_ORIG_ALLOC]] {ssbuffer.block_id = 5 : i32} : memref<16xf32> to tensor<16xf32>
// Cloned alloc + to_tensor at consumer block (block_id = 50).
// CHECK:       %[[G_CLONE_ALLOC:.*]] = memref.alloc() {ssbuffer.block_id = 50 : i32} : memref<16xf32>
// CHECK:       %[[G_CLONE_TT:.*]] = bufferization.to_tensor %[[G_CLONE_ALLOC]] {ssbuffer.block_id = 50 : i32} : memref<16xf32> to tensor<16xf32>
// CHECK:       arith.addf %[[G_CLONE_TT]], %[[G_CLONE_TT]] {ssbuffer.block_id = 50 : i32} : tensor<16xf32>
// NO UB alloc, NO copy chain.
// CHECK-NOT:   memref.alloc() : memref<16xf32, #hivm.address_space<ub>>
// CHECK-NOT:   hivm.hir.copy{{.*}}: tensor<16xf32>
// Constant 1 carrying block_id = 50 (relocated with the addi, follows the
// first user which is the clone).
// CHECK:       %{{.+}} = arith.constant {ssbuffer.block_id = 50 : i32} 1 : i32
// Counter add-1 carries block_id = 50 and is tagged iterCounter.
// CHECK:       %{{.+}} = arith.addi %{{.+}}, %{{.+}} {ssbuffer.block_id = 50 : i32, ssbuffer.iterCounter} : i32
// scf.yield forwards the counter (referenced by the addi's %addi-res), proving
// SSA is preserved.
// CHECK:       scf.yield %{{.*}}, %{{.*}}, %{{.+}}
// Counter-aware whileOp carries ssbuffer.iterCounter alongside main_loop.
// CHECK:       } {{.*}}ssbuffer.iterCounter, {{.*}}ssbuffer.main_loop = 1 : i64

// ---- Inputs ----

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">,
                   ssbuffer.intra_buf_count = 1 : i32} {
  // T-while-A: whileOp with main_loop on the terminator, single-buffer.
  func.func @test_while_mainloop_bufnum_one() {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128xf32>
    %carry = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0_i32)
          : (tensor<128xf32>, i32) -> (tensor<128xf32>, i32) {
        %cmp = arith.cmpi slt, %arg1, %c10_i32 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : tensor<128xf32>, i32
      } do {
      ^bb0(%arg0: tensor<128xf32>, %arg1: i32):
        // Producer block_id = 7.
        %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xf32>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 7 : i32} : memref<128xf32> to tensor<128xf32>
        // Consumer block_id = 10 (cross-block).
        %consumed = arith.addf %prod, %prod {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
        %next = arith.addi %arg1, %c1_i32 : i32
        scf.yield %consumed, %next : tensor<128xf32>, i32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 23 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">,
                   ssbuffer.intra_buf_count = 2 : i32} {
  // T-while-B: whileOp with main_loop on the terminator, multi-buffer (bufNum=2).
  func.func @test_while_mainloop_bufnum_two() {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<128xf32>
    %carry = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<128xf32>) -> tensor<128xf32>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0_i32)
          : (tensor<128xf32>, i32) -> (tensor<128xf32>, i32) {
        %cmp = arith.cmpi slt, %arg1, %c10_i32 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : tensor<128xf32>, i32
      } do {
      ^bb0(%arg0: tensor<128xf32>, %arg1: i32):
        %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xf32>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 7 : i32} : memref<128xf32> to tensor<128xf32>
        %consumed = arith.addf %prod, %prod {ssbuffer.block_id = 10 : i32} : tensor<128xf32>
        %next = arith.addi %arg1, %c1_i32 : i32
        scf.yield %consumed, %next : tensor<128xf32>, i32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 23 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // T-while-C: whileOp with main_loop on the OP itself (not terminator).
  func.func @test_while_mainloop_attr_on_op() {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_one = arith.constant 1.0 : f16
    %init = tensor.empty() : tensor<64xf16>
    %carry = linalg.fill ins(%cst_one : f16) outs(%init : tensor<64xf16>) -> tensor<64xf16>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0_i32)
          : (tensor<64xf16>, i32) -> (tensor<64xf16>, i32) {
        %cmp = arith.cmpi slt, %arg1, %c10_i32 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : tensor<64xf16>, i32
      } do {
      ^bb0(%arg0: tensor<64xf16>, %arg1: i32):
        %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<64xf16>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 7 : i32} : memref<64xf16> to tensor<64xf16>
        // Multi-region consumer (scf.if inside the do-region).
        %cnd = arith.cmpi slt, %arg1, %c1_i32 {ssbuffer.block_id = 11 : i32} : i32
        %consumed = scf.if %cnd -> (tensor<64xf16>) {
          %a = arith.addf %prod, %prod {ssbuffer.block_id = 11 : i32} : tensor<64xf16>
          scf.yield %a : tensor<64xf16>
        } else {
          %b = arith.mulf %prod, %prod {ssbuffer.block_id = 12 : i32} : tensor<64xf16>
          scf.yield %b : tensor<64xf16>
        } {ssbuffer.block_id = 12 : i32}
        %next = arith.addi %arg1, %c1_i32 : i32
        scf.yield %consumed, %next : tensor<64xf16>, i32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 23 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // T-while-D: whileOp main_loop with cross-block dep that tests
  // getOutermostSsbufferId's main_loop-first priority.
  // Producer at block_id=8, consumer at block_id=12, both inside do-region.
  // whileOp itself is at block_id=5 (so the buggy code would have collapsed
  // both to 5, hiding the cross-block dep).
  func.func @test_while_outermost_id_priority() {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<32xf32>
    %carry = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<32xf32>) -> tensor<32xf32>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0_i32)
          : (tensor<32xf32>, i32) -> (tensor<32xf32>, i32) {
        %cmp = arith.cmpi slt, %arg1, %c10_i32 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : tensor<32xf32>, i32
      } do {
      ^bb0(%arg0: tensor<32xf32>, %arg1: i32):
        %alloc = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32xf32>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 8 : i32} : memref<32xf32> to tensor<32xf32>
        %consumed = arith.addf %prod, %prod {ssbuffer.block_id = 12 : i32} : tensor<32xf32>
        %next = arith.addi %arg1, %c1_i32 : i32
        scf.yield %consumed, %next : tensor<32xf32>, i32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 5 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // T-while-E: whileOp main_loop with empty+fill pattern.
  // Two pairs of empty+fill, both cloned to consumer block 12.
  func.func @test_while_clone_empty_fill() {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_one = arith.constant 1.0 : f32
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %c0_i32, %arg1 = %cst_one)
          : (i32, f32) -> (i32, f32) {
        %cmp = arith.cmpi slt, %arg0, %c10_i32 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : i32, f32
      } do {
      ^bb0(%arg0: i32, %arg1: f32):
        // Producer block_id = 8: empty+fill (cloneable).
        %empty8 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<32x1xf32>
        %fill8 = linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst_one : f32)
            outs(%empty8 : tensor<32x1xf32>) -> tensor<32x1xf32>
        // Producer block_id = 14: empty+fill (cloneable).
        %empty14 = tensor.empty() {ssbuffer.block_id = 14 : i32} : tensor<32x1xf32>
        %fill14 = linalg.fill {ssbuffer.block_id = 14 : i32} ins(%cst_one : f32)
            outs(%empty14 : tensor<32x1xf32>) -> tensor<32x1xf32>
        // Consumer block_id = 12: uses BOTH fills cross-block.
        %consumed = arith.addf %fill8, %fill14 {ssbuffer.block_id = 12 : i32} : tensor<32x1xf32>
        %next = arith.addi %arg0, %c1_i32 : i32
        scf.yield %next, %arg1 : i32, f32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 23 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // T-while-G: dedicated regression test for relocateWhileCounterOps.
  // Producer at block_id = 5, consumer at block_id = 50, whileOp at block_id = 99.
  // Before the fix: counter add-1 would land at block_id = 99 (whileOp's own id).
  // After the fix: counter add-1 (and its constant 1) land at block_id = 5
  // (the block_id of the first op that consumes the counter iter_arg).
  func.func @test_while_counter_relocation() {
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_zero = arith.constant 0.0 : f32
    %init = tensor.empty() : tensor<16xf32>
    %carry = linalg.fill ins(%cst_zero : f32) outs(%init : tensor<16xf32>) -> tensor<16xf32>
    scope.scope : () -> () {
      %result:2 = scf.while (%arg0 = %carry, %arg1 = %c0_i32)
          : (tensor<16xf32>, i32) -> (tensor<16xf32>, i32) {
        %cmp = arith.cmpi slt, %arg1, %c10_i32 {ssbuffer.block_id = 16 : i32} : i32
        scf.condition(%cmp) %arg0, %arg1 : tensor<16xf32>, i32
      } do {
      ^bb0(%arg0: tensor<16xf32>, %arg1: i32):
        // Producer block_id = 5.
        %alloc = memref.alloc() {ssbuffer.block_id = 5 : i32} : memref<16xf32>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 5 : i32} : memref<16xf32> to tensor<16xf32>
        // Consumer block_id = 50 (very different from 5, to make the relocation
        // verdict unambiguous).
        %consumed = arith.addf %prod, %prod {ssbuffer.block_id = 50 : i32} : tensor<16xf32>
        %next = arith.addi %arg1, %c1_i32 : i32
        scf.yield %consumed, %next : tensor<16xf32>, i32
      } attributes {ssbuffer.main_loop = 1 : i64, ssbuffer.block_id = 99 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
