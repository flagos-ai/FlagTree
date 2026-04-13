// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-fence-insertion=compute-capability=90 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @insert_for_wgmma_shared_operand
  tt.func @insert_for_wgmma_shared_operand(
      %a: tensor<64x512xbf16, #dot>,
      %b_init: tensor<512x64xbf16, #blocked>) -> tensor<64x64xf32, #mma> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>

    // CHECK: %[[B_SMEM:.+]] = ttg.local_alloc
    %b_smem = ttg.local_alloc %b_init : (tensor<512x64xbf16, #blocked>) -> !ttg.memdesc<512x64xbf16, #shared, #smem, mutable>

    // CHECK-NEXT: tle.wgmma_shared_operand_fence %[[B_SMEM]]
    // CHECK-NEXT: ttng.warp_group_dot %arg0, %[[B_SMEM]], {{.*}}
    %out = ttng.warp_group_dot %a, %b_smem, %acc {inputPrecision = 0 : i32} : tensor<64x512xbf16, #dot> * !ttg.memdesc<512x64xbf16, #shared, #smem, mutable> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }
}
