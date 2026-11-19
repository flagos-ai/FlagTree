// RUN: triton-opt --clone-ops --verify-diagnostics %s --allow-unregistered-dialect

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
        %1 = linalg.fill {ssbuffer.block_id = 11 : i32} ins(%c1_f32 : f32) outs(%0 : tensor<256x64xf32>) -> tensor<256x64xf32>
        %2 = arith.addf %1, %t1 {ssbuffer.block_id = 12 : i32} : tensor<256x64xf32>
        %3 = linalg.fill {ssbuffer.block_id = 12 : i32} ins(%c1_f32 : f32) outs(%2 : tensor<256x64xf32>) -> tensor<256x64xf32>
      } {ssbuffer.block_id = 10 : i32, ssbuffer.main_loop = 0 : i32}
      scope.return
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>}
    return {ssbuffer.core_type = "VECTOR"}
  }
}
