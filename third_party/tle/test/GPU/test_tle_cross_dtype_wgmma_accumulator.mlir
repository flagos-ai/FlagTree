// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: triton-opt %s --split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' --convert-nv-gpu-to-llvm | FileCheck %s

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

// -----

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
  // CHECK-LABEL: @fp8_k32_acc_to_bf16_k16_reg_a_unmarked
  // CHECK: wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
  // CHECK-NOT: wgmma.mma_async.sync.aligned.m64n64k32.f32.bf16.bf16
  tt.func @fp8_k32_acc_to_bf16_k16_reg_a_unmarked(
      %a: tensor<64x64xbf16, #dot_bf16>,
      %b: !ttg.memdesc<64x64xbf16, #shared_b_bf16, #smem>,
      %acc: tensor<64x64xf32, #mma_fp8>) {
    %res = ttng.warp_group_dot %a, %b, %acc {
      inputPrecision = 0 : i32,
      isAsync = true
    } : tensor<64x64xbf16, #dot_bf16> * !ttg.memdesc<64x64xbf16, #shared_b_bf16, #smem> -> tensor<64x64xf32, #mma_fp8>
    tt.return
  }

}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#sa = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#sb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @same_dtype_bf16
  // CHECK: wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
  tt.func @same_dtype_bf16(%a: !ttg.memdesc<64x64xbf16, #sa, #smem>, %b: !ttg.memdesc<64x64xbf16, #sb, #smem>, %acc: tensor<64x64xf32, #mma>) {
    %res = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 2147483647 : i32} : !ttg.memdesc<64x64xbf16, #sa, #smem> * !ttg.memdesc<64x64xbf16, #sb, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#sa = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#sb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @fp8_acc_to_f16
  // CHECK: wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
  tt.func @fp8_acc_to_f16(%a: !ttg.memdesc<64x64xf16, #sa, #smem>, %b: !ttg.memdesc<64x64xf16, #sb, #smem>, %acc: tensor<64x64xf32, #mma>) {
    %res = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 2147483647 : i32} : !ttg.memdesc<64x64xf16, #sa, #smem> * !ttg.memdesc<64x64xf16, #sb, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#sa = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#sb = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @bf16_acc_to_fp8
  // CHECK: wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e5m2
  tt.func @bf16_acc_to_fp8(%a: !ttg.memdesc<64x64xf8E4M3FN, #sa, #smem>, %b: !ttg.memdesc<64x64xf8E5M2, #sb, #smem>, %acc: tensor<64x64xf32, #mma>) {
    %res = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 2147483647 : i32} : !ttg.memdesc<64x64xf8E4M3FN, #sa, #smem> * !ttg.memdesc<64x64xf8E5M2, #sb, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#sa = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#sb = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @bf16_acc_to_tf32
  // CHECK: wgmma.mma_async.sync.aligned.m64n64k8.f32.tf32.tf32
  tt.func @bf16_acc_to_tf32(%a: !ttg.memdesc<64x64xf32, #sa, #smem>, %b: !ttg.memdesc<64x64xf32, #sb, #smem>, %acc: tensor<64x64xf32, #mma>) {
    %res = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 2147483647 : i32} : !ttg.memdesc<64x64xf32, #sa, #smem> * !ttg.memdesc<64x64xf32, #sb, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#sa = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 8}>
#sb = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 8}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.target" = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @same_dtype_int8
  // CHECK: wgmma.mma_async.sync.aligned.m64n64k32.s32.s8.s8
  tt.func @same_dtype_int8(%a: !ttg.memdesc<64x64xi8, #sa, #smem>, %b: !ttg.memdesc<64x64xi8, #sb, #smem>, %acc: tensor<64x64xi32, #mma>) {
    %res = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 2147483647 : i32} : !ttg.memdesc<64x64xi8, #sa, #smem> * !ttg.memdesc<64x64xi8, #sb, #smem> -> tensor<64x64xi32, #mma>
    tt.return
  }
}
