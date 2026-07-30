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

// RUN: triton-opt %s -split-input-file -pass-pipeline='builtin.module(convert-triton-gpu-to-llvm{compute-capability=90})' | FileCheck %s --check-prefix=NOMOVE
// RUN: triton-opt %s -split-input-file -pass-pipeline='builtin.module(convert-triton-gpu-to-llvm{compute-capability=90})' | FileCheck %s --check-prefix=REGS

// A pure register relabel: concatenating two K=64 dot-operand fragments into one
// K=128 fragment must move registers and nothing else. The two run lines are
// separate because CHECK-NOT only holds until the next positive match, so the
// forbidden ops and the register counts cannot share a prefix.

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // No shared memory, barrier or mma anywhere in the function.
  // NOMOVE-LABEL: llvm.func @concat_dot_fragments_dot_operand
  // NOMOVE-NOT: llvm.load
  // NOMOVE-NOT: llvm.store
  // NOMOVE-NOT: nvvm.barrier0
  // NOMOVE-NOT: nvg.wgmma
  // NOMOVE-NOT: llvm.inline_asm
  // NOMOVE: llvm.return

  // Every result register is filled from a source register: 128 slots in, 128 out.
  // REGS-LABEL: llvm.func @concat_dot_fragments_dot_operand
  // REGS-COUNT-128: llvm.extractvalue
  // REGS-COUNT-128: llvm.insertvalue
  // REGS: llvm.return
  tt.func @concat_dot_fragments_dot_operand(%a0: tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>,
                                %a1: tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) {
    %merged = tle.concat_dot_fragments %a0, %a1 {dim = 1 : i32} : tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, tensor<128x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> -> tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    tt.return
  }
}

// -----

// The contraction axis is derived from the rank, so a batched fragment merges
// along its last dim just the same.

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [1, 4, 1], instrShape = [1, 16, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // NOMOVE-LABEL: llvm.func @concat_dot_fragments_batched
  // NOMOVE-NOT: llvm.load
  // NOMOVE-NOT: llvm.store
  // NOMOVE-NOT: nvvm.barrier0
  // NOMOVE: llvm.return

  // REGS-LABEL: llvm.func @concat_dot_fragments_batched
  // REGS-COUNT-64: llvm.extractvalue
  // REGS-COUNT-64: llvm.insertvalue
  // REGS: llvm.return
  tt.func @concat_dot_fragments_batched(%a0: tensor<1x128x32xbf16, #dot0>, %a1: tensor<1x128x32xbf16, #dot0>) {
    %merged = tle.concat_dot_fragments %a0, %a1 {dim = 2 : i32} : tensor<1x128x32xbf16, #dot0>, tensor<1x128x32xbf16, #dot0> -> tensor<1x128x64xbf16, #dot0>
    tt.return
  }
}
