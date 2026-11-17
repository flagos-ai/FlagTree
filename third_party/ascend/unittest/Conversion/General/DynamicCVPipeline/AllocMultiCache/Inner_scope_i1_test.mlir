// RUN: triton-opt --add_multi_buffer_inner_scope %s | FileCheck %s
// Verifies that the i1 tensor dep triggers the IGNORED (rc=2) fallback path,
// not the FAILED (rc=1) path. i1 cross-block tensor deps are intentionally
// not multi-buffered (the multi-buffer pipeline assumes memref-shaped
// tensors with element types the downstream pipeline can clone), so the
// pass must signal IGNORED (rc=2 = ERRCODE_IGNORED) rather than FAILED
// (rc=1 = ERRCODE_FAILED).

// CHECK-LABEL: module attributes
// CHECK-SAME: triton_ascend.dynamic_cv_pipeline.rc = 2 : i32
// Confirm we did NOT clobber the IGNORED code with FAILED.
// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc = 1

// CHECK-LABEL: func.func @test_i1_tensor_dep_multi_buffer

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_i1_tensor_dep_multi_buffer() {
    %c0_i32 = arith.constant 0 : i32
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst_true = arith.constant true
    scope.scope : () -> () {
      scf.for %i = %c0_i32 to %c100_i32 step %c1_i32  : i32 {
        // Producer: tensor<128xi1> in block 7
        %alloc = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<128xi1>
        %prod = bufferization.to_tensor %alloc {ssbuffer.block_id = 7 : i32} : memref<128xi1> to tensor<128xi1>
        // Consumer in block 10 (cross-block)
        %consumed = arith.ori %prod, %prod {ssbuffer.block_id = 10 : i32} : tensor<128xi1>
      } {ssbuffer.main_loop = 1 : i64}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return
  }
}
