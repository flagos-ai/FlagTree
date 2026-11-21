// RUN: triton-opt --clone-ops --allow-unregistered-dialect %s | FileCheck %s

// Unit test for CloneOps pass: `tensor.empty` is a pure metadata op that
// allocates a tensor shape but holds no data. When it is cloned into a
// later block within a VECTOR main_loop, the cloned `tensor.empty` must be
// allowed (the validator skips it when checking for tensor-typed results).
// Without this carve-out, `tensor.empty` would be flagged as a tensor-typed
// result and the pass would signal a failure.

// CHECK-LABEL: func.func @test_vector_clone_tensor_op
// The cloned `tensor.empty` (clone of the block-11 `tensor.empty`) must be
// present in block 12 with the `clone = 11` attribute. If the validator
// incorrectly rejected it, the IR would either be missing this op or the
// pass would signal a failure.
// CHECK: tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.clone = 11 : i32} : tensor<256x64xf32>
// The pre-existing block-12 tensor-typed ops (which are NOT clones) must
// remain untouched and have no `clone` attribute.
// CHECK: arith.addf {{%[0-9]+, %[0-9]+}} {ssbuffer.block_id = 12 : i32} : tensor<256x64xf32>
// CHECK-NOT: arith.addf {{.*}}ssbuffer.clone
// CHECK: linalg.fill {ssbuffer.block_id = 12 : i32}

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_vector_clone_tensor_op(%arg0: memref<?xi8>, %arg1: memref<?xi8>) attributes {global_kernel = "local", mix_mode = "mix", parallel_mode = "simd"} {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c10_i64 = arith.constant 10 : i64
    %c1_f32 = arith.constant 1.000000e+00 : f32
    %t1 = tensor.empty() : tensor<256x64xf32>
    scope.scope : () -> () {
      %0 = tensor.empty() : tensor<256x64xf32>
      scf.for %arg2 = %c0_i64 to %c10_i64 step %c1_i64  : i64 {
        %1 = tensor.empty() {ssbuffer.block_id = 11 : i32} : tensor<256x64xf32>
        %2 = arith.addf %1, %t1 {ssbuffer.block_id = 12 : i32} : tensor<256x64xf32>
        %3 = linalg.fill {ssbuffer.block_id = 12 : i32} ins(%c1_f32 : f32) outs(%2 : tensor<256x64xf32>) -> tensor<256x64xf32>
      } {ssbuffer.block_id = 10 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
