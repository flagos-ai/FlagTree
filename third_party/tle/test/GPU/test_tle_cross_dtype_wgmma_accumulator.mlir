// Copyright 2025- FlagOS Contributors
//
// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' --convert-nv-gpu-to-llvm | FileCheck %s

#mma_fp8 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#mma_bf16 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot_bf16 = #ttg.dot_op<{opIdx = 0, parent = #mma_bf16, kWidth = 2}>
#shared_a_bf16 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b_bf16 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The async C value retains the preceding FP8/K32 MMA encoding. The current
  // BF16 register-A operand selects K16, while both encodings have identical C
  // register ownership.
  // CHECK-LABEL: @fp8_k32_acc_to_bf16_k16_reg_a
  // CHECK: wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
  // CHECK-NOT: wgmma.mma_async.sync.aligned.m64n64k32.f32.bf16.bf16
  tt.func @fp8_k32_acc_to_bf16_k16_reg_a(
      %a: tensor<64x64xbf16, #dot_bf16>,
      %b: !ttg.memdesc<64x64xbf16, #shared_b_bf16, #smem>,
      %acc: tensor<64x64xf32, #mma_fp8>) {
    %res = ttng.warp_group_dot %a, %b, %acc {
      inputPrecision = 0 : i32,
      isAsync = true,
      tle.wgmma_accumulator_chain_c
    } : tensor<64x64xbf16, #dot_bf16> * !ttg.memdesc<64x64xbf16, #shared_b_bf16, #smem> -> tensor<64x64xf32, #mma_fp8>
    tt.return
  }

  // Shared-A has no BF16 dot-parent encoding, and the production scheduling
  // pipeline may no longer carry the chain marker by this point. Current BF16
  // operands still select K16 while preserving FP8 C's M/N ownership.
  // CHECK-LABEL: @fp8_k32_acc_to_bf16_k16_shared_shared
  // CHECK: wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
  // CHECK-NOT: wgmma.mma_async.sync.aligned.m64n64k32.f32.bf16.bf16
  tt.func @fp8_k32_acc_to_bf16_k16_shared_shared(
      %a: !ttg.memdesc<64x64xbf16, #shared_a_bf16, #smem>,
      %b: !ttg.memdesc<64x64xbf16, #shared_b_bf16, #smem>,
      %acc: tensor<64x64xf32, #mma_fp8>) {
    %res = ttng.warp_group_dot %a, %b, %acc {
      inputPrecision = 0 : i32,
      isAsync = true
    } : !ttg.memdesc<64x64xbf16, #shared_a_bf16, #smem> * !ttg.memdesc<64x64xbf16, #shared_b_bf16, #smem> -> tensor<64x64xf32, #mma_fp8>
    tt.return
  }
}
