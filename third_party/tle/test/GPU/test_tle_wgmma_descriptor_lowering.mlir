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

// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' | FileCheck %s --check-prefix=NVGPU
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' --convert-nv-gpu-to-llvm | FileCheck %s --check-prefix=LLVM

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // NVGPU-LABEL: @descriptor_arithmetic_remains_visible_to_ptxas
  // NVGPU: nvg.wgmma
  // NVGPU-SAME: tle.wgmma_operand_a_desc_imm
  // NVGPU-SAME: tle.wgmma_operand_b_desc_imm
  // LLVM-LABEL: @descriptor_arithmetic_remains_visible_to_ptxas
  // LLVM: llvm.inline_asm
  // LLVM-SAME: add.u64 __tle_wgmma_desc_a
  // LLVM-SAME: add.u64 __tle_wgmma_desc_b
  // LLVM-SAME: wgmma.mma_async.sync.aligned
  // LLVM-SAME: __tle_wgmma_desc_a, __tle_wgmma_desc_b
  tt.func @descriptor_arithmetic_remains_visible_to_ptxas(
      %a: !ttg.memdesc<64x64xf16, #shared_a, #smem>,
      %b: !ttg.memdesc<64x64xf16, #shared_b, #smem>,
      %acc: tensor<64x64xf32, #mma>) {
    %m = ttng.warp_group_dot %a, %b, %acc { inputPrecision = 0 : i32 }:
      !ttg.memdesc<64x64xf16, #shared_a, #smem> * !ttg.memdesc<64x64xf16, #shared_b, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // NVGPU-LABEL: @native_active_n_for_tiled_rows
  // NVGPU-COUNT-1: nvg.wgmma
  // NVGPU-SAME: n = 80 : i32
  // LLVM-LABEL: @native_active_n_for_tiled_rows
  // LLVM: llvm.inline_asm
  // LLVM-SAME: 0x4000004000010000
  // LLVM-SAME: wgmma.mma_async.sync.aligned.m64n80k16
  // LLVM-NOT: wgmma.mma_async.sync.aligned.m64n16k16
  tt.func @native_active_n_for_tiled_rows(
      %a: !ttg.memdesc<64x256xf16, #shared_a, #smem>,
      %b: !ttg.memdesc<256x128xf16, #shared_b, #smem>,
      %acc: tensor<64x128xf32, #mma>) {
    %m = ttng.warp_group_dot %a, %b, %acc {
      inputPrecision = 0 : i32,
      tle.tiled_smem_logical_cols = 256 : i32,
      tle.tiled_smem_logical_rows = 80 : i32,
      tle.tiled_smem_operand_b,
      tle.tiled_smem_storage_tile_shape = array<i32: 16, 64>,
      tle.wgmma_active_n = 80 : i32
    } : !ttg.memdesc<64x256xf16, #shared_a, #smem> * !ttg.memdesc<256x128xf16, #shared_b, #smem> -> tensor<64x128xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The plan alone identifies compact B; no legacy tiled metadata is needed.
  // NVGPU-LABEL: @planned_transposed_view
  // NVGPU: nvg.wgmma
  // NVGPU-SAME: n = 80 : i32
  // LLVM-LABEL: @planned_transposed_view
  // LLVM: llvm.inline_asm
  // LLVM-SAME: 0x4000004000010000
  // LLVM-SAME: wgmma.mma_async.sync.aligned.m64n80k16
  // LLVM-NOT: wgmma.mma_async.sync.aligned.m64n16k16
  tt.func @planned_transposed_view(
      %a: !ttg.memdesc<64x256xf16, #shared_a, #smem>,
      %b: !ttg.memdesc<256x128xf16, #shared_b, #smem>,
      %acc: tensor<64x128xf32, #mma>) {
    %m = ttng.warp_group_dot %a, %b, %acc {
      inputPrecision = 0 : i32,
      tle.wgmma_operand_b_plan = {
        storage = {version = 1 : i32, shape = array<i64: 80, 256>,
                   tile = array<i64: 16, 64>, encoding = #shared_a},
        view_transposed = true, instruction_n = 80 : i32,
        lbo_bytes = 16 : i64, sbo_bytes = 1024 : i64
      },
      tle.wgmma_active_n = 80 : i32
    } : !ttg.memdesc<64x256xf16, #shared_a, #smem> * !ttg.memdesc<256x128xf16, #shared_b, #smem> -> tensor<64x128xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#storage = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // A direct view of K-contiguous storage has the same K-major B
  // descriptor. Its carrier encoding does not define the planned storage.
  // NVGPU-LABEL: @planned_direct_view_of_transposed_storage
  // NVGPU: nvg.wgmma
  // NVGPU-SAME: n = 80 : i32
  // LLVM-LABEL: @planned_direct_view_of_transposed_storage
  // LLVM: llvm.inline_asm
  // LLVM-SAME: 0x4000004000010000
  // LLVM-SAME: wgmma.mma_async.sync.aligned.m64n80k16
  // LLVM-NOT: wgmma.mma_async.sync.aligned.m64n16k16
  tt.func @planned_direct_view_of_transposed_storage(
      %a: !ttg.memdesc<64x256xf16, #shared_a, #smem>,
      %b: !ttg.memdesc<256x128xf16, #shared_a, #smem>,
      %acc: tensor<64x128xf32, #mma>) {
    %m = ttng.warp_group_dot %a, %b, %acc {
      inputPrecision = 0 : i32,
      tle.wgmma_operand_b_plan = {
        storage = {version = 1 : i32, shape = array<i64: 256, 80>,
                   tile = array<i64: 64, 16>, encoding = #storage},
        view_transposed = false, instruction_n = 80 : i32,
        lbo_bytes = 16 : i64, sbo_bytes = 1024 : i64
      },
      tle.wgmma_active_n = 80 : i32
    } : !ttg.memdesc<64x256xf16, #shared_a, #smem> * !ttg.memdesc<256x128xf16, #shared_a, #smem> -> tensor<64x128xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // A plan must also select logical lowering without any active extent attr.
  // NVGPU-LABEL: @planned_full_extent
  // NVGPU: nvg.wgmma
  // NVGPU-SAME: n = 64 : i32
  // LLVM-LABEL: @planned_full_extent
  // LLVM: llvm.inline_asm
  // LLVM-SAME: 0x4000004000010000
  // LLVM-SAME: wgmma.mma_async.sync.aligned.m64n64k16
  tt.func @planned_full_extent(
      %a: !ttg.memdesc<64x64xf16, #shared, #smem>,
      %b: !ttg.memdesc<64x64xf16, #shared, #smem>,
      %acc: tensor<64x64xf32, #mma>) {
    %m = ttng.warp_group_dot %a, %b, %acc {
      inputPrecision = 0 : i32,
      tle.wgmma_operand_b_plan = {
        storage = {version = 1 : i32, shape = array<i64: 64, 64>,
                   tile = array<i64: 16, 64>, encoding = #shared},
        view_transposed = true, instruction_n = 64 : i32,
        lbo_bytes = 16 : i64, sbo_bytes = 1024 : i64
      }
    } : !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}
