// RUN: triton-opt %s -split-input-file --tritongpu-optimize-dot-operands | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 8}>
#shared2 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @keep_fp8_transposed_wgmma_a_staging_alloc
  tt.func @keep_fp8_transposed_wgmma_a_staging_alloc(
      %b: !ttg.memdesc<64x128xf8E4M3FN, #shared2, #smem, mutable>) -> tensor<128x128xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #mma>

    %a_smem = ttg.local_alloc : () -> !ttg.memdesc<1x64x128xf8E4M3FN, #shared, #smem, mutable>
    %a_slot = ttg.memdesc_index %a_smem[%c0] : !ttg.memdesc<1x64x128xf8E4M3FN, #shared, #smem, mutable> -> !ttg.memdesc<64x128xf8E4M3FN, #shared, #smem, mutable>
    %a = ttg.local_load %a_slot : !ttg.memdesc<64x128xf8E4M3FN, #shared, #smem, mutable> -> tensor<64x128xf8E4M3FN, #blocked>
    %a_t = tt.trans %a {order = array<i32: 1, 0>} : tensor<64x128xf8E4M3FN, #blocked> -> tensor<128x64xf8E4M3FN, #blocked1>

    // CHECK-NOT: tle.memdesc_wgmma_view
    // CHECK: %[[A_ALLOC:.+]] = ttg.local_alloc {{.*}} : (tensor<128x64xf8E4M3FN
    // CHECK-NOT: tle.memdesc_wgmma_view
    // CHECK: %[[A_DOT:.+]] = ttg.local_load %[[A_ALLOC]]
    // CHECK-NOT: tle.memdesc_wgmma_view
    // CHECK: ttng.warp_group_dot %[[A_DOT]], %arg0, {{.*}}
    %a_alloc = ttg.local_alloc %a_t : (tensor<128x64xf8E4M3FN, #blocked1>) -> !ttg.memdesc<128x64xf8E4M3FN, #shared1, #smem>
    %a_dot = ttg.local_load %a_alloc : !ttg.memdesc<128x64xf8E4M3FN, #shared1, #smem> -> tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>>
    %out = ttng.warp_group_dot %a_dot, %b, %acc {inputPrecision = 0 : i32, maxNumImpreciseAcc = 1073741824 : i32} : tensor<128x64xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 4}>> * !ttg.memdesc<64x128xf8E4M3FN, #shared2, #smem, mutable> -> tensor<128x128xf32, #mma>
    tt.return %out : tensor<128x128xf32, #mma>
  }
}
