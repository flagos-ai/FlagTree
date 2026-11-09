// RUN: triton-opt --add_multi_buffer_inner_scope %s | FileCheck %s

// T-cloned-fill-scalar-dep: empty+fill is cloned into the consumer block. Three
// cases are checked:
//
// Case A (test_cloned_fill_scalar_dep_same_parent):
//   `ins` is %extracted = tensor.extract %reduced[], defined inside the same
//   parentOp (the main_loop forOp body) as the tensor.empty. All three ops
//   (tensor.empty, tensor.extract, linalg.fill) must be cloned together so the
//   cloned fill's `ins` references the cloned tensor.extract, not the
//   producer-side %extracted.
//
// Case B (test_cloned_fill_scalar_dep_diff_parent):
//   `ins` is %cst = arith.constant, defined OUTSIDE the main_loop (different
//   parentOp from tensor.empty). Only tensor.empty + linalg.fill should be
//   cloned; the arith.constant is a BlockArgument-like outer-scope value and
//   must not be cloned.
//
// Case C (test_clone_introduces_new_tensor_dep):
//   Exercises the two-phase dep collection: `ins` is tensor.extract of a
//   producer-side tensor (%T = linalg.fill with block_id=13) defined INSIDE
//   main_loop. Phase 1 clones empty+fill; the cloned extract then reads %T
//   from a consumer block (block_id=9), creating a NEW cross-block tensor
//   ref. Phase 2 must re-collect deps and add a multi-buffer for %T. The
//   cloned extract must read from the multi-buffer result, not %T directly.

// CHECK-LABEL: func.func @test_cloned_fill_scalar_dep_same_parent
// CHECK: %[[ORIG_EXTRACT:.*]] = tensor.extract {{.*}} {ssbuffer.block_id = 13 : i32}
// CHECK: %[[ORIG_FILL:.*]] = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%[[ORIG_EXTRACT]] : f32) outs({{.*}}) -> tensor<32x1xf32>
// CHECK: %[[CLONE_EXTRACT:.*]] = tensor.extract {{.*}} {ssbuffer.block_id = 9 : i32}
// CHECK: %[[CLONE_FILL:.*]] = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%[[CLONE_EXTRACT]] : f32) outs({{.*}}) -> tensor<32x1xf32>

// CHECK-LABEL: func.func @test_cloned_fill_scalar_dep_diff_parent
// CHECK: %[[ORIG_FILL2:.*]] = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%{{.*}} : f32) outs({{.*}}) -> tensor<32x1xf32>
// CHECK: %[[CLONE_FILL2:.*]] = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%{{.*}} : f32) outs({{.*}}) -> tensor<32x1xf32>
// CHECK-NOT: arith.constant {{.*}}{ssbuffer.block_id = 9 : i32}

// Case C: cloned fill's ins reaches producer-side %T → multi-buffer for %T.
// Multi-buffer allocs at top of main_loop scope (ping/pong memrefs).
// CHECK-LABEL: func.func @test_clone_introduces_new_dep
// CHECK-DAG: memref.alloc() : memref<f32, #hivm.address_space<ub>>
// CHECK-DAG: memref.alloc() : memref<f32, #hivm.address_space<ub>>
// Producer intraDeps tag moved from memref.memory_space_cast to hivm.hir.copy
// (refactor innerscope): the cast is just a buffer container, the copy is the
// actual producer behavior op.
// CHECK-NOT: memref.memory_space_cast{{.+}}intraDeps
// Producer-side ping-pong into multi-buffer (block_id = 13, intra_buffer).
// Producer scf.if has no result type (consumer does: -> tensor<f32>).
// CHECK: scf.if %{{[0-9_a-zA-Z]+}} {
// Producer-side hivm.hir.copy now carries intraDeps = [0, 1] (refactor innerscope).
// CHECK:   hivm.hir.copy ins(%{{.+}} : tensor<f32>) outs(%{{.+}} : memref<f32>) {{.+}}intraDeps = [0 : i32, 1 : i32]
// CHECK: } {ssbuffer.block_id = 13 : i32, ssbuffer.intra_buffer}
// Original empty+fill remains in producer block (block_id = 13).
// CHECK: linalg.fill {ssbuffer.block_id = 13 : i32} ins(%extracted : f32) outs({{.*}} : tensor<32x1xf32>) -> tensor<32x1xf32>
// Consumer-side multi-buffer selection (block_id = 22 = inner loop body).
// CHECK: scf.if {{.*}} -> (tensor<f32>) {
// CHECK:   bufferization.to_tensor {{.*}} restrict writable : memref<f32>
// CHECK: } else {
// CHECK:   bufferization.to_tensor {{.*}} restrict writable : memref<f32>
// CHECK: } {ssbuffer.block_id = 22 : i32, {{.*}}ssbuffer.intra_buffer}
// Cloned extract reads multi-buffer result (NOT %1 directly).
// CHECK: %[[C_CLONE_EXTRACT:.*]] = tensor.extract %{{.*}}[] {ssbuffer.block_id = 9 : i32} : tensor<f32>
// Cloned fill in consumer block: ins references the cloned extract.
// CHECK: linalg.fill {ssbuffer.block_id = 9 : i32} ins(%[[C_CLONE_EXTRACT]] : f32) outs({{.*}} : tensor<32x1xf32>) -> tensor<32x1xf32>

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // Case A: ins defining op (tensor.extract) shares parentOp with tensor.empty
  func.func @test_cloned_fill_scalar_dep_same_parent(%arg0: memref<?xf16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %cst = arith.constant 1.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %reduced_init = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<f32>
    %reduced = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%reduced_init : tensor<f32>) -> tensor<f32>
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
        %empty = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<32x1xf32>
        %extracted = tensor.extract %reduced[] {ssbuffer.block_id = 13 : i32} : tensor<f32>
        %fill = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%extracted : f32) outs(%empty : tensor<32x1xf32>) -> tensor<32x1xf32>
        scf.for %j = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
          %34 = tensor.empty() {ssbuffer.block_id = 9 : i32} : tensor<32x1xf32>
          %35 = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst : f32) outs(%34 : tensor<32x1xf32>) -> tensor<32x1xf32>
          %36 = arith.mulf %35, %fill {ssbuffer.block_id = 9 : i32} : tensor<32x1xf32>
        } {Undefined, ssbuffer.block_id = 22 : i32}
      } {Undefined, ssbuffer.block_id = 23 : i32, ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // Case B: ins defining op (arith.constant) is OUTSIDE main_loop, so only
  // tensor.empty + linalg.fill are cloned.
  func.func @test_cloned_fill_scalar_dep_diff_parent(%arg0: memref<?xf16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %cst = arith.constant {ssbuffer.block_id = 15 : i32} 1.0 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
        %empty = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<32x1xf32>
        %fill = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%empty : tensor<32x1xf32>) -> tensor<32x1xf32>
        scf.for %j = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
          %34 = tensor.empty() {ssbuffer.block_id = 9 : i32} : tensor<32x1xf32>
          %35 = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst : f32) outs(%34 : tensor<32x1xf32>) -> tensor<32x1xf32>
          %36 = arith.mulf %35, %fill {ssbuffer.block_id = 9 : i32} : tensor<32x1xf32>
        } {Undefined, ssbuffer.block_id = 22 : i32}
      } {Undefined, ssbuffer.block_id = 23 : i32, ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }

  // Case C: empty+fill's `ins` is tensor.extract of a producer-side tensor
  // defined INSIDE main_loop. Phase 1 clones the empty+fill, the cloned
  // extract creates a NEW cross-block ref to %T (block_id=13), and Phase 2
  // must add a multi-buffer for %T.
  func.func @test_clone_introduces_new_dep(%arg0: memref<?xf16>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c100_i32 = arith.constant 100 : i32
    %cst = arith.constant 1.0 : f32
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
        %T_alloc = bufferization.alloc_tensor() {ssbuffer.block_id = 13 : i32} : tensor<f32>
        %T = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%cst : f32) outs(%T_alloc : tensor<f32>) -> tensor<f32>
        %extracted = tensor.extract %T[] {ssbuffer.block_id = 13 : i32} : tensor<f32>
        %empty = tensor.empty() {ssbuffer.block_id = 13 : i32} : tensor<32x1xf32>
        %fill = linalg.fill {ssbuffer.block_id = 13 : i32} ins(%extracted : f32) outs(%empty : tensor<32x1xf32>) -> tensor<32x1xf32>
        scf.for %j = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
          %inner_empty = tensor.empty() {ssbuffer.block_id = 9 : i32} : tensor<32x1xf32>
          %inner_fill = linalg.fill {ssbuffer.block_id = 9 : i32} ins(%cst : f32) outs(%inner_empty : tensor<32x1xf32>) -> tensor<32x1xf32>
          %consumed = arith.mulf %inner_fill, %fill {ssbuffer.block_id = 9 : i32} : tensor<32x1xf32>
        } {Undefined, ssbuffer.block_id = 22 : i32}
      } {Undefined, ssbuffer.block_id = 23 : i32, ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
