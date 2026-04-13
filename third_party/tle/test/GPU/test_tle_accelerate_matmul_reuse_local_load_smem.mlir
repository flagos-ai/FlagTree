// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reuse_wgmma_b_from_local_load
  tt.func @reuse_wgmma_b_from_local_load(
      %a: tensor<64x64xbf16, #blocked>,
      %b_init: tensor<64x512xbf16, #blocked1>) -> tensor<64x512xf32, #blocked1> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x512xf32, #blocked1>

    // CHECK: %[[B_SMEM:.+]] = ttg.local_alloc %{{.*}} : (tensor<64x512xbf16, #blocked1>) -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    %b_smem = ttg.local_alloc %b_init : (tensor<64x512xbf16, #blocked1>) -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>

    %b = ttg.local_load %b_smem : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> tensor<64x512xbf16, #blocked1>

    // CHECK-NOT: ttg.local_load
    // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<64x512xbf16
    // CHECK: ttng.warp_group_dot {{.*}}, %[[B_SMEM]], {{.*}} : {{.*}} * !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> tensor<64x512xf32
    %a_dot = ttg.convert_layout %a : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    %b_dot = ttg.convert_layout %b : tensor<64x512xbf16, #blocked1> -> tensor<64x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
    %out = tt.dot %a_dot, %b_dot, %acc : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<64x512xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<64x512xf32, #blocked1>
    tt.return %out : tensor<64x512xf32, #blocked1>
  }
}
