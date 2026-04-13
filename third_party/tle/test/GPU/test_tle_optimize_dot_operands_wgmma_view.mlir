// RUN: triton-opt %s -split-input-file --tritongpu-optimize-dot-operands | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reuse_transposed_wgmma_b_from_existing_smem
  tt.func @reuse_transposed_wgmma_b_from_existing_smem(
      %a: tensor<64x512xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>,
      %b_init: tensor<64x512xbf16, #blocked>) -> tensor<64x64xf32, #mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    // CHECK: %[[B_SMEM:.+]] = ttg.local_alloc
    %b_smem = ttg.local_alloc %b_init : (tensor<64x512xbf16, #blocked>) -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    %b = ttg.local_load %b_smem : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> tensor<64x512xbf16, #blocked>
    %b_t = tt.trans %b {order = array<i32: 1, 0>} : tensor<64x512xbf16, #blocked> -> tensor<512x64xbf16, #blocked1>

    // CHECK: %[[VIEW:.+]] = tle.memdesc_wgmma_view %[[B_SMEM]] {order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
    // CHECK-NOT: ttg.local_alloc {{.*}} : (tensor<512x64xbf16
    // CHECK: ttng.warp_group_dot {{.*}}, %[[VIEW]], {{.*}}
    %b_alloc = ttg.local_alloc %b_t : (tensor<512x64xbf16, #blocked1>) -> !ttg.memdesc<512x64xbf16, #shared, #smem>
    %out = ttng.warp_group_dot %a, %b_alloc, %acc {inputPrecision = 0 : i32} : tensor<64x512xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<512x64xbf16, #shared, #smem> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }

  // CHECK-LABEL: tt.func @reuse_wgmma_a_from_existing_smem
  tt.func @reuse_wgmma_a_from_existing_smem(
      %a_init: tensor<64x512xbf16, #blocked>,
      %b: !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>) -> tensor<64x64xf32, #mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    // CHECK: %[[A_SMEM:.+]] = ttg.local_alloc
    %a_smem = ttg.local_alloc %a_init : (tensor<64x512xbf16, #blocked>) -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
    %a = ttg.local_load %a_smem : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> tensor<64x512xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>

    // CHECK: ttng.warp_group_dot %[[A_SMEM]], %arg1, {{.*}} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
    %out = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32} : tensor<64x512xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }
}
