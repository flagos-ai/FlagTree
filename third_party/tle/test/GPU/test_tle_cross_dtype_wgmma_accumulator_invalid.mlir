// Copyright 2025- FlagOS Contributors
//
// RUN: not triton-opt %s --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' 2>&1 | FileCheck %s

#mma_fp8 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#mma_bf16_bad_n = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#dot_bf16_bad_n = #ttg.dot_op<{opIdx = 0, parent = #mma_bf16_bad_n, kWidth = 2}>
#shared_bf16 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: error: cannot reuse an async WGMMA accumulator across incompatible physical C layouts
  tt.func @reject_incompatible_accumulator_n_layout(
      %a: tensor<64x64xbf16, #dot_bf16_bad_n>,
      %b: !ttg.memdesc<64x64xbf16, #shared_bf16, #smem>,
      %acc: tensor<64x64xf32, #mma_fp8>) {
    %res = ttng.warp_group_dot %a, %b, %acc {
      inputPrecision = 0 : i32,
      isAsync = true,
      tle.wgmma_accumulator_chain_c
    } : tensor<64x64xbf16, #dot_bf16_bad_n> * !ttg.memdesc<64x64xbf16, #shared_bf16, #smem> -> tensor<64x64xf32, #mma_fp8>
    tt.return
  }
}
