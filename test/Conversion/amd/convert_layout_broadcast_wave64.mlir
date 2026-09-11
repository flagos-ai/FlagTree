// FlagTree regression coverage for triton/pull/11646.
// Adapt the FlagTree #1047 pointer case to AMD wave64.
// RUN: triton-opt %s --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx942 --cse | FileCheck %s

// The 64-bit pointer conversion should use two lane shuffles.
#src = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [64, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#dst = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @convert_broadcast_wave64_pointer
  tt.func @convert_broadcast_wave64_pointer(%arg0: tensor<16x1x!tt.ptr<f32>, #src>) {
    // CHECK-NOT: rocdl.s.barrier
    // CHECK-NOT: llvm.store
    // CHECK-COUNT-2: rocdl.ds_bpermute
    // CHECK-NOT: rocdl.ds_bpermute
    // CHECK-NOT: rocdl.s.barrier
    // CHECK-NOT: llvm.load
    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<16x1x!tt.ptr<f32>, #src> -> tensor<16x1x!tt.ptr<f32>, #dst>
    tt.return
  }
}
