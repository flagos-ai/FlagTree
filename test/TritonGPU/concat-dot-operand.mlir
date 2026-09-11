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

// RUN: triton-opt %s -split-input-file -tritongpu-concat-dot-operand | FileCheck %s --check-prefixes=COMMON,MATCH
// RUN: triton-opt %s -split-input-file -tritongpu-concat-dot-operand -tritongpu-expand-concat-dot-operand | FileCheck %s --check-prefixes=COMMON,ROUNDTRIP
// RUN: triton-opt %s -split-input-file -tritongpu-merge-segmented-dot | FileCheck %s --check-prefix=MERGE
// RUN: triton-opt %s -split-input-file -tritongpu-merge-segmented-dot -tritongpu-remove-layout-conversions -tritongpu-merge-segmented-dot | FileCheck %s --check-prefix=SPLIT

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#j = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#t = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>
#lin = #ttg.linear<{register = [[0, 4]], lane = [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>

// The smallest instance of the shape a kernel writes: join -> trans -> reshape
// is an ordered concat along K, so fold it into one op. The convert in between
// is the state the pass sees, since it runs before the operand layouts are
// assigned.
// MATCH-LABEL: @concat_two_a_side
// MATCH-NOT: tt.join
// MATCH: ttg.concat_dot_operand %arg0, %arg1 {dim = 1 : i32}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @concat_two_a_side(%a: tensor<8x4xf16, #blocked>, %b: tensor<8x4xf16, #blocked>,
                             %rhs: tensor<8x8xf16, #dB>, %acc: tensor<8x8xf32, #mma>)
      -> tensor<8x8xf32, #mma> {
    %j = tt.join %a, %b : tensor<8x4xf16, #blocked> -> tensor<8x4x2xf16, #j>
    %tr = tt.trans %j {order = array<i32: 0, 2, 1>} : tensor<8x4x2xf16, #j> -> tensor<8x2x4xf16, #t>
    %r = tt.reshape %tr : tensor<8x2x4xf16, #t> -> tensor<8x8xf16, #lin>
    %c = ttg.convert_layout %r : tensor<8x8xf16, #lin> -> tensor<8x8xf16, #dA>
    %d = tt.dot %c, %rhs, %acc : tensor<8x8xf16, #dA> * tensor<8x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d : tensor<8x8xf32, #mma>
  }
}

// A non-power-of-two total K cannot be represented by one dot-operand layout.
// The greedy segmented rewrite still merges the largest representable prefix:
// 16 + 16 becomes 32, while the final 16-wide segment remains separate.
// MERGE-LABEL: @merge_three_equal_mmav2
// MERGE: ttg.concat_dot_operand %arg0, %arg1 {dim = 1 : i32}
// MERGE: ttg.concat_dot_operand %arg3, %arg4 {dim = 0 : i32}
// MERGE-COUNT-2: tt.dot
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @merge_three_equal_mmav2(%a0: tensor<16x16xf16, #dA>, %a1: tensor<16x16xf16, #dA>, %a2: tensor<16x16xf16, #dA>,
                                   %b0: tensor<16x8xf16, #dB>, %b1: tensor<16x8xf16, #dB>, %b2: tensor<16x8xf16, #dB>,
                                   %acc: tensor<16x8xf32, #mma>) -> tensor<16x8xf32, #mma> {
    %d0 = tt.dot %a0, %b0, %acc : tensor<16x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<16x8xf32, #mma>
    %d1 = tt.dot %a1, %b1, %d0 : tensor<16x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<16x8xf32, #mma>
    %d2 = tt.dot %a2, %b2, %d1 : tensor<16x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<16x8xf32, #mma>
    tt.return %d2 : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// No join tree is present here: this is the IR shape produced when a lowering
// emits independent K fragments and accumulates them with a dot chain. The
// generic segmented pass still folds both operands into one dot, including
// heterogeneous static K segments (16 + 16 + 32 + 32 + 32).
// MERGE-LABEL: @merge_direct_mmav2
// MERGE: ttg.concat_dot_operand %arg0, %arg1, %arg2, %arg3, %arg4 {dim = 1 : i32}
// MERGE: ttg.concat_dot_operand %arg5, %arg6, %arg7, %arg8, %arg9 {dim = 0 : i32}
// MERGE: tt.dot
// MERGE-NEXT: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @merge_direct_mmav2(%a0: tensor<16x16xf16, #dA>, %a1: tensor<16x16xf16, #dA>, %a2: tensor<16x32xf16, #dA>, %a3: tensor<16x32xf16, #dA>, %a4: tensor<16x32xf16, #dA>,
                              %b0: tensor<16x8xf16, #dB>, %b1: tensor<16x8xf16, #dB>, %b2: tensor<32x8xf16, #dB>, %b3: tensor<32x8xf16, #dB>, %b4: tensor<32x8xf16, #dB>,
                              %acc: tensor<16x8xf32, #mma>) -> tensor<16x8xf32, #mma> {
    %d0 = tt.dot %a0, %b0, %acc : tensor<16x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<16x8xf32, #mma>
    %d1 = tt.dot %a1, %b1, %d0 : tensor<16x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<16x8xf32, #mma>
    %d2 = tt.dot %a2, %b2, %d1 : tensor<16x32xf16, #dA> * tensor<32x8xf16, #dB> -> tensor<16x8xf32, #mma>
    %d3 = tt.dot %a3, %b3, %d2 : tensor<16x32xf16, #dA> * tensor<32x8xf16, #dB> -> tensor<16x8xf32, #mma>
    %d4 = tt.dot %a4, %b4, %d3 : tensor<16x32xf16, #dA> * tensor<32x8xf16, #dB> -> tensor<16x8xf32, #mma>
    tt.return %d4 : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#bb = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#bj = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 8, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#bt = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [1, 4, 8], warpsPerCTA = [1, 1, 1], order = [0, 2, 1]}>
#blin = #ttg.linear<{register = [[4, 0]], lane = [[0, 1], [0, 2], [0, 4], [1, 0], [2, 0]], warp = [], block = []}>

// The B side concatenates along dim 0, which is the contraction axis for
// opIdx 1. The permutation and the growing dim both move, so this is the case
// that pins `isConcatOrder` against a hardcoded axis.
// MATCH-LABEL: @concat_two_b_side
// MATCH-NOT: tt.join
// MATCH: ttg.concat_dot_operand %arg0, %arg1 {dim = 0 : i32}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @concat_two_b_side(%a: tensor<4x8xf16, #bb>, %b: tensor<4x8xf16, #bb>,
                             %lhs: tensor<8x8xf16, #dA>, %acc: tensor<8x8xf32, #mma>)
      -> tensor<8x8xf32, #mma> {
    %j = tt.join %a, %b : tensor<4x8xf16, #bb> -> tensor<4x8x2xf16, #bj>
    %tr = tt.trans %j {order = array<i32: 2, 0, 1>} : tensor<4x8x2xf16, #bj> -> tensor<2x4x8xf16, #bt>
    %r = tt.reshape %tr : tensor<2x4x8xf16, #bt> -> tensor<8x8xf16, #blin>
    %c = ttg.convert_layout %r : tensor<8x8xf16, #blin> -> tensor<8x8xf16, #dB>
    %d = tt.dot %lhs, %c, %acc : tensor<8x8xf16, #dA> * tensor<8x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d : tensor<8x8xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#j = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#j4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [8, 4, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [3, 2, 1, 0]}>
#t4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [8, 1, 1, 4], warpsPerCTA = [1, 1, 1, 1], order = [1, 2, 3, 0]}>
#lin4 = #ttg.linear<{register = [[0, 8], [0, 4]], lane = [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// Four fragments. The segment count is whatever the tree depth says; nothing in
// the rewrite is specialised to two.
// MATCH-LABEL: @concat_four
// MATCH-NOT: tt.join
// MATCH: ttg.concat_dot_operand %arg0, %arg1, %arg2, %arg3 {dim = 1 : i32}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @concat_four(%a0: tensor<8x4xf16, #blocked>, %a1: tensor<8x4xf16, #blocked>,
                       %a2: tensor<8x4xf16, #blocked>, %a3: tensor<8x4xf16, #blocked>,
                       %rhs: tensor<16x8xf16, #dB>, %acc: tensor<8x8xf32, #mma>)
      -> tensor<8x8xf32, #mma> {
    %j0 = tt.join %a0, %a1 : tensor<8x4xf16, #blocked> -> tensor<8x4x2xf16, #j>
    %j1 = tt.join %a2, %a3 : tensor<8x4xf16, #blocked> -> tensor<8x4x2xf16, #j>
    %q = tt.join %j0, %j1 : tensor<8x4x2xf16, #j> -> tensor<8x4x2x2xf16, #j4>
    %tr = tt.trans %q {order = array<i32: 0, 3, 2, 1>} : tensor<8x4x2x2xf16, #j4> -> tensor<8x2x2x4xf16, #t4>
    %r = tt.reshape %tr : tensor<8x2x2x4xf16, #t4> -> tensor<8x16xf16, #lin4>
    %c = ttg.convert_layout %r : tensor<8x16xf16, #lin4> -> tensor<8x16xf16, #dA>
    %d = tt.dot %c, %rhs, %acc : tensor<8x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d : tensor<8x8xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// A kernel is free to build its K tile in stages -- stack pairs, then stack the
// results -- which the recognizer folds into nested concats. Flattening them is
// what makes the staged spelling reach the same single wide operand as the
// one-shot one.
// MATCH-LABEL: @flatten_staged_concats
// MATCH: ttg.concat_dot_operand %arg0, %arg1, %arg2, %arg3 {dim = 1 : i32}
// MATCH-SAME: -> tensor<8x16xf16
// MATCH-NOT: ttg.concat_dot_operand
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @flatten_staged_concats(%a0: tensor<8x4xf16, #blocked>, %a1: tensor<8x4xf16, #blocked>,
                                  %a2: tensor<8x4xf16, #blocked>, %a3: tensor<8x4xf16, #blocked>,
                                  %rhs: tensor<16x8xf16, #dB>, %acc: tensor<8x8xf32, #mma>)
      -> tensor<8x8xf32, #mma> {
    %s0 = ttg.concat_dot_operand %a0, %a1 {dim = 1 : i32} : tensor<8x4xf16, #blocked>, tensor<8x4xf16, #blocked> -> tensor<8x8xf16, #blocked>
    %s1 = ttg.concat_dot_operand %a2, %a3 {dim = 1 : i32} : tensor<8x4xf16, #blocked>, tensor<8x4xf16, #blocked> -> tensor<8x8xf16, #blocked>
    %w = ttg.concat_dot_operand %s0, %s1 {dim = 1 : i32} : tensor<8x8xf16, #blocked>, tensor<8x8xf16, #blocked> -> tensor<8x16xf16, #blocked>
    %c = ttg.convert_layout %w : tensor<8x16xf16, #blocked> -> tensor<8x16xf16, #dA>
    %d = tt.dot %c, %rhs, %acc : tensor<8x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d : tensor<8x8xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// A stage that also feeds something else stays live, so flattening it would
// recompute rather than replace it. The chain is left nested and each level
// lowers on its own.
// MATCH-LABEL: @keep_escaping_stage
// MATCH-COUNT-3: ttg.concat_dot_operand
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @keep_escaping_stage(%a0: tensor<8x4xf16, #blocked>, %a1: tensor<8x4xf16, #blocked>,
                               %a2: tensor<8x4xf16, #blocked>, %a3: tensor<8x4xf16, #blocked>,
                               %rhs: tensor<16x8xf16, #dB>, %acc: tensor<8x8xf32, #mma>)
      -> (tensor<8x8xf32, #mma>, tensor<8x8xf16, #blocked>) {
    %s0 = ttg.concat_dot_operand %a0, %a1 {dim = 1 : i32} : tensor<8x4xf16, #blocked>, tensor<8x4xf16, #blocked> -> tensor<8x8xf16, #blocked>
    %s1 = ttg.concat_dot_operand %a2, %a3 {dim = 1 : i32} : tensor<8x4xf16, #blocked>, tensor<8x4xf16, #blocked> -> tensor<8x8xf16, #blocked>
    %w = ttg.concat_dot_operand %s0, %s1 {dim = 1 : i32} : tensor<8x8xf16, #blocked>, tensor<8x8xf16, #blocked> -> tensor<8x16xf16, #blocked>
    %c = ttg.convert_layout %w : tensor<8x16xf16, #blocked> -> tensor<8x16xf16, #dA>
    %d = tt.dot %c, %rhs, %acc : tensor<8x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d, %s0 : tensor<8x8xf32, #mma>, tensor<8x8xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#j = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#lin2 = #ttg.linear<{register = [[0, 1]], lane = [[0, 2], [0, 4], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>

// A transpose that is not the concat order leaves the chain alone: flattening it
// would interleave the fragments instead of concatenating them.
// COMMON-LABEL: @wrong_order
// COMMON-NOT: ttg.concat_dot_operand
// COMMON: tt.join
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wrong_order(%a: tensor<8x4xf16, #blocked>, %b: tensor<8x4xf16, #blocked>)
      -> tensor<8x8xf16, #lin2> {
    %j = tt.join %a, %b : tensor<8x4xf16, #blocked> -> tensor<8x4x2xf16, #j>
    %tr = tt.trans %j {order = array<i32: 0, 1, 2>} : tensor<8x4x2xf16, #j> -> tensor<8x4x2xf16, #j>
    %r = tt.reshape %tr : tensor<8x4x2xf16, #j> -> tensor<8x8xf16, #lin2>
    tt.return %r : tensor<8x8xf16, #lin2>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#j = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#t = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>

// An allow_reorder reshape says nothing about element order, so the chain cannot
// be proven to be an ordered concat.
// COMMON-LABEL: @allow_reorder
// COMMON-NOT: ttg.concat_dot_operand
// COMMON: tt.join
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @allow_reorder(%a: tensor<8x4xf16, #blocked>, %b: tensor<8x4xf16, #blocked>)
      -> tensor<8x8xf16, #blocked> {
    %j = tt.join %a, %b : tensor<8x4xf16, #blocked> -> tensor<8x4x2xf16, #j>
    %tr = tt.trans %j {order = array<i32: 0, 2, 1>} : tensor<8x4x2xf16, #j> -> tensor<8x2x4xf16, #t>
    %r = tt.reshape %tr allow_reorder : tensor<8x2x4xf16, #t> -> tensor<8x8xf16, #blocked>
    tt.return %r : tensor<8x8xf16, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#j = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#t = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>
#lin = #ttg.linear<{register = [[0, 4]], lane = [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>

// The join also feeds a second consumer, so folding the chain would leave it live
// and duplicate the work.
// COMMON-LABEL: @join_escapes
// COMMON-NOT: ttg.concat_dot_operand
// COMMON: tt.join
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @join_escapes(%a: tensor<8x4xf16, #blocked>, %b: tensor<8x4xf16, #blocked>)
      -> (tensor<8x8xf16, #lin>, tensor<8x4x2xf16, #j>) {
    %j = tt.join %a, %b : tensor<8x4xf16, #blocked> -> tensor<8x4x2xf16, #j>
    %tr = tt.trans %j {order = array<i32: 0, 2, 1>} : tensor<8x4x2xf16, #j> -> tensor<8x2x4xf16, #t>
    %r = tt.reshape %tr : tensor<8x2x4xf16, #t> -> tensor<8x8xf16, #lin>
    tt.return %r, %j : tensor<8x8xf16, #lin>, tensor<8x4x2xf16, #j>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#j = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#t = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>
#lin = #ttg.linear<{register = [[0, 4]], lane = [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>

// The result is only stored, so it never gets a dot_op layout and the relabel
// could not lower; leave the chain alone rather than break a legal program.
// COMMON-LABEL: @not_a_dot_operand
// COMMON-NOT: ttg.concat_dot_operand
// COMMON: tt.join
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @not_a_dot_operand(%a: tensor<8x4xf16, #blocked>, %b: tensor<8x4xf16, #blocked>,
                             %out: tensor<8x8x!tt.ptr<f16>, #lin>) {
    %j = tt.join %a, %b : tensor<8x4xf16, #blocked> -> tensor<8x4x2xf16, #j>
    %tr = tt.trans %j {order = array<i32: 0, 2, 1>} : tensor<8x4x2xf16, #j> -> tensor<8x2x4xf16, #t>
    %r = tt.reshape %tr : tensor<8x2x4xf16, #t> -> tensor<8x8xf16, #lin>
    tt.store %out, %r : tensor<8x8x!tt.ptr<f16>, #lin>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#j = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#t = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>
#lin = #ttg.linear<{register = [[0, 4]], lane = [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// Reaching only the accumulator says nothing about how a K tile is split, and an
// accumulator never carries a dot_op encoding, so a concat built for it could
// never be relabeled and would only have to be expanded again.
// COMMON-LABEL: @only_reaches_accumulator
// COMMON-NOT: ttg.concat_dot_operand
// COMMON: tt.join
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @only_reaches_accumulator(%a: tensor<8x4xf32, #blocked>, %b: tensor<8x4xf32, #blocked>,
                                    %lhs: tensor<8x8xf16, #dA>, %rhs: tensor<8x8xf16, #dB>)
      -> tensor<8x8xf32, #mma> {
    %j = tt.join %a, %b : tensor<8x4xf32, #blocked> -> tensor<8x4x2xf32, #j>
    %tr = tt.trans %j {order = array<i32: 0, 2, 1>} : tensor<8x4x2xf32, #j> -> tensor<8x2x4xf32, #t>
    %r = tt.reshape %tr : tensor<8x2x4xf32, #t> -> tensor<8x8xf32, #lin>
    %c = ttg.convert_layout %r : tensor<8x8xf32, #lin> -> tensor<8x8xf32, #mma>
    %d = tt.dot %lhs, %rhs, %c : tensor<8x8xf16, #dA> * tensor<8x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d : tensor<8x8xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#j = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#j4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [8, 4, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [3, 2, 1, 0]}>
#t4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [8, 1, 1, 4], warpsPerCTA = [1, 1, 1, 1], order = [1, 2, 3, 0]}>
#lin4 = #ttg.linear<{register = [[0, 8], [0, 4]], lane = [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// A join below the root also feeds something else, so folding would leave it
// live and recompute the same values.
// COMMON-LABEL: @inner_join_escapes
// COMMON-NOT: ttg.concat_dot_operand
// COMMON: tt.join
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @inner_join_escapes(%a0: tensor<8x4xf16, #blocked>, %a1: tensor<8x4xf16, #blocked>,
                              %a2: tensor<8x4xf16, #blocked>, %a3: tensor<8x4xf16, #blocked>,
                              %rhs: tensor<16x8xf16, #dB>, %acc: tensor<8x8xf32, #mma>)
      -> (tensor<8x8xf32, #mma>, tensor<8x4x2xf16, #j>) {
    %j0 = tt.join %a0, %a1 : tensor<8x4xf16, #blocked> -> tensor<8x4x2xf16, #j>
    %j1 = tt.join %a2, %a3 : tensor<8x4xf16, #blocked> -> tensor<8x4x2xf16, #j>
    %q = tt.join %j0, %j1 : tensor<8x4x2xf16, #j> -> tensor<8x4x2x2xf16, #j4>
    %tr = tt.trans %q {order = array<i32: 0, 3, 2, 1>} : tensor<8x4x2x2xf16, #j4> -> tensor<8x2x2x4xf16, #t4>
    %r = tt.reshape %tr : tensor<8x2x2x4xf16, #t4> -> tensor<8x16xf16, #lin4>
    %c = ttg.convert_layout %r : tensor<8x16xf16, #lin4> -> tensor<8x16xf16, #dA>
    %d = tt.dot %c, %rhs, %acc : tensor<8x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d, %j0 : tensor<8x8xf32, #mma>, tensor<8x4x2xf16, #j>
  }
}

// -----

#b = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#b3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#b4 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1], threadsPerWarp = [8, 4, 1, 1], warpsPerCTA = [1, 1, 1, 1], order = [3, 2, 1, 0]}>
#b5 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1], threadsPerWarp = [8, 4, 1, 1, 1], warpsPerCTA = [1, 1, 1, 1, 1], order = [4, 3, 2, 1, 0]}>
#t5 = #ttg.blocked<{sizePerThread = [1, 1, 1, 1, 1], threadsPerWarp = [8, 1, 1, 1, 4], warpsPerCTA = [1, 1, 1, 1, 1], order = [1, 2, 3, 4, 0]}>
#lin = #ttg.linear<{register = [[0, 16], [0, 8], [0, 4]], lane = [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// Eight fragments: the shape a real dequantizing kernel writes, one per packed
// word. Folding it and expanding it again must restore the same chain -- the
// operands here keep a blocked layout, which the relabel cannot use, so the
// expansion runs and has to be an exact inverse of the match.
// MATCH-LABEL: @fold_then_expand
// MATCH: ttg.concat_dot_operand %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7 {dim = 1 : i32}
// ROUNDTRIP-LABEL: @fold_then_expand
// ROUNDTRIP-COUNT-7: tt.join
// ROUNDTRIP: tt.trans {{.*}} {order = array<i32: 0, 4, 3, 2, 1>}
// ROUNDTRIP-NOT: ttg.concat_dot_operand
// ROUNDTRIP: tt.dot
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fold_then_expand(%a0: tensor<8x4xf16, #b>, %a1: tensor<8x4xf16, #b>,
                            %a2: tensor<8x4xf16, #b>, %a3: tensor<8x4xf16, #b>,
                            %a4: tensor<8x4xf16, #b>, %a5: tensor<8x4xf16, #b>,
                            %a6: tensor<8x4xf16, #b>, %a7: tensor<8x4xf16, #b>,
                            %rhs: tensor<32x8xf16, #dB>, %acc: tensor<8x8xf32, #mma>)
      -> tensor<8x8xf32, #mma> {
    %j0 = tt.join %a0, %a1 : tensor<8x4xf16, #b> -> tensor<8x4x2xf16, #b3>
    %j1 = tt.join %a2, %a3 : tensor<8x4xf16, #b> -> tensor<8x4x2xf16, #b3>
    %j2 = tt.join %a4, %a5 : tensor<8x4xf16, #b> -> tensor<8x4x2xf16, #b3>
    %j3 = tt.join %a6, %a7 : tensor<8x4xf16, #b> -> tensor<8x4x2xf16, #b3>
    %p0 = tt.join %j0, %j1 : tensor<8x4x2xf16, #b3> -> tensor<8x4x2x2xf16, #b4>
    %p1 = tt.join %j2, %j3 : tensor<8x4x2xf16, #b3> -> tensor<8x4x2x2xf16, #b4>
    %q = tt.join %p0, %p1 : tensor<8x4x2x2xf16, #b4> -> tensor<8x4x2x2x2xf16, #b5>
    %tr = tt.trans %q {order = array<i32: 0, 4, 3, 2, 1>} : tensor<8x4x2x2x2xf16, #b5> -> tensor<8x2x2x2x4xf16, #t5>
    %r = tt.reshape %tr : tensor<8x2x2x2x4xf16, #t5> -> tensor<8x32xf16, #lin>
    %c = ttg.convert_layout %r : tensor<8x32xf16, #lin> -> tensor<8x32xf16, #dA>
    %d = tt.dot %c, %rhs, %acc : tensor<8x32xf16, #dA> * tensor<32x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d : tensor<8x8xf32, #mma>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// A flattened concat is expanded as one balanced tree over all four fragments,
// not as the two stages it was built from. That is the point of flattening: the
// staged and one-shot spellings become the same op and take the same path.
// ROUNDTRIP-LABEL: @expand_flattened_concat
// ROUNDTRIP-COUNT-3: tt.join
// ROUNDTRIP-NOT: tt.join
// ROUNDTRIP: tt.trans {{.*}} {order = array<i32: 0, 3, 2, 1>}
// ROUNDTRIP-NOT: ttg.concat_dot_operand
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @expand_flattened_concat(%a0: tensor<8x4xf16, #blocked>, %a1: tensor<8x4xf16, #blocked>,
                                   %a2: tensor<8x4xf16, #blocked>, %a3: tensor<8x4xf16, #blocked>,
                                   %rhs: tensor<16x8xf16, #dB>, %acc: tensor<8x8xf32, #mma>)
      -> tensor<8x8xf32, #mma> {
    %s0 = ttg.concat_dot_operand %a0, %a1 {dim = 1 : i32} : tensor<8x4xf16, #blocked>, tensor<8x4xf16, #blocked> -> tensor<8x8xf16, #blocked>
    %s1 = ttg.concat_dot_operand %a2, %a3 {dim = 1 : i32} : tensor<8x4xf16, #blocked>, tensor<8x4xf16, #blocked> -> tensor<8x8xf16, #blocked>
    %w = ttg.concat_dot_operand %s0, %s1 {dim = 1 : i32} : tensor<8x8xf16, #blocked>, tensor<8x8xf16, #blocked> -> tensor<8x16xf16, #blocked>
    %c = ttg.convert_layout %w : tensor<8x16xf16, #blocked> -> tensor<8x16xf16, #dA>
    %d = tt.dot %c, %rhs, %acc : tensor<8x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d : tensor<8x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// The relabel needs at least 8 * kWidth along the contraction axis: below that
// the dot_op layout replicates elements across lanes instead of scaling with K,
// so a fragment is not a K-slice of the wider operand. A kernel that packs 8
// values per word and tiles BLOCK_K=64 lands exactly here, so this must fall
// back cleanly rather than relabel the wrong registers.
// ROUNDTRIP-LABEL: @expand_below_kwidth_floor
// ROUNDTRIP: tt.join
// ROUNDTRIP-NOT: ttg.concat_dot_operand
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @expand_below_kwidth_floor(%a0: tensor<16x8xf16, #dA>, %a1: tensor<16x8xf16, #dA>,
                                     %rhs: tensor<16x8xf16, #dB>, %acc: tensor<16x8xf32, #mma>)
      -> tensor<16x8xf32, #mma> {
    %w = ttg.concat_dot_operand %a0, %a1 {dim = 1 : i32} : tensor<16x8xf16, #dA>, tensor<16x8xf16, #dA> -> tensor<16x16xf16, #dA>
    %d = tt.dot %w, %rhs, %acc : tensor<16x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<16x8xf32, #mma>
    tt.return %d : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// A concat that already carries the operand layout is a pure register relabel
// and must survive to the lowering: expanding it would put a join tree back in
// front of the mma for nothing.
// ROUNDTRIP-LABEL: @keep_relabelable_concat
// ROUNDTRIP-NOT: tt.join
// ROUNDTRIP: ttg.concat_dot_operand
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @keep_relabelable_concat(%a0: tensor<16x16xf16, #dA>, %a1: tensor<16x16xf16, #dA>,
                                   %rhs: tensor<32x8xf16, #dB>, %acc: tensor<16x8xf32, #mma>)
      -> tensor<16x8xf32, #mma> {
    %w = ttg.concat_dot_operand %a0, %a1 {dim = 1 : i32} : tensor<16x16xf16, #dA>, tensor<16x16xf16, #dA> -> tensor<16x32xf16, #dA>
    %d = tt.dot %w, %rhs, %acc : tensor<16x32xf16, #dA> * tensor<32x8xf16, #dB> -> tensor<16x8xf32, #mma>
    tt.return %d : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// `concat(extract(w, 0), extract(w, 1))` is `w`. The extracts only become
// recognizable as a complete cover once layout propagation has unified their
// sources, which is after the recognizer has run, so this is a canonicalization
// rather than a pattern private to one pass.
// MERGE-LABEL: @fold_concat_of_extracts
// MERGE-NOT: ttg.extract_dot_operand
// MERGE-NOT: ttg.concat_dot_operand
// MERGE: tt.dot %arg0, %arg1, %arg2
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @fold_concat_of_extracts(%w: tensor<16x32xf16, #dA>, %rhs: tensor<32x8xf16, #dB>,
                                   %acc: tensor<16x8xf32, #mma>) -> tensor<16x8xf32, #mma> {
    %e0 = ttg.extract_dot_operand %w {dim = 1 : i32, index = 0 : i32} : tensor<16x32xf16, #dA> -> tensor<16x16xf16, #dA>
    %e1 = ttg.extract_dot_operand %w {dim = 1 : i32, index = 1 : i32} : tensor<16x32xf16, #dA> -> tensor<16x16xf16, #dA>
    %c = ttg.concat_dot_operand %e0, %e1 {dim = 1 : i32} : tensor<16x16xf16, #dA>, tensor<16x16xf16, #dA> -> tensor<16x32xf16, #dA>
    %d = tt.dot %c, %rhs, %acc : tensor<16x32xf16, #dA> * tensor<32x8xf16, #dB> -> tensor<16x8xf32, #mma>
    tt.return %d : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// Extracts out of order are not a cover of the source in operand order, so the
// fold must not fire: `concat(extract(w,1), extract(w,0))` is a K swap, not `w`.
// MERGE-LABEL: @keep_concat_of_reordered_extracts
// MERGE: ttg.extract_dot_operand
// MERGE: ttg.concat_dot_operand
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @keep_concat_of_reordered_extracts(%w: tensor<16x32xf16, #dA>, %rhs: tensor<32x8xf16, #dB>,
                                             %acc: tensor<16x8xf32, #mma>) -> tensor<16x8xf32, #mma> {
    %e0 = ttg.extract_dot_operand %w {dim = 1 : i32, index = 1 : i32} : tensor<16x32xf16, #dA> -> tensor<16x16xf16, #dA>
    %e1 = ttg.extract_dot_operand %w {dim = 1 : i32, index = 0 : i32} : tensor<16x32xf16, #dA> -> tensor<16x16xf16, #dA>
    %c = ttg.concat_dot_operand %e0, %e1 {dim = 1 : i32} : tensor<16x16xf16, #dA>, tensor<16x16xf16, #dA> -> tensor<16x32xf16, #dA>
    %d = tt.dot %c, %rhs, %acc : tensor<16x32xf16, #dA> * tensor<32x8xf16, #dB> -> tensor<16x8xf32, #mma>
    tt.return %d : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#b3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#t3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>

// The inverse view: a kernel that reshape/transpose/splits one wide operand
// into per-segment dots. Recognising the split tree turns the leaves into
// extracts off one wide value; the chain matcher then concatenates them back
// and the concat-of-extracts fold collapses the round trip, so the dot reads
// the value it was split from and the tree is gone.
//
// The split rewrite has to run before the chain matcher, or the leaves stop
// being dot operands and the tree survives with a concat stacked on top of it.
// SPLIT-LABEL: @split_tree_round_trip
// SPLIT-NOT: tt.split
// SPLIT-NOT: ttg.extract_dot_operand
// SPLIT-NOT: ttg.concat_dot_operand %{{.*}} {dim = 1 : i32}
// SPLIT: %[[A:.*]] = ttg.convert_layout %arg0
// SPLIT: tt.dot %[[A]]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @split_tree_round_trip(%w: tensor<16x32xf16, #blocked>, %r0: tensor<16x8xf16, #dB>,
                                 %r1: tensor<16x8xf16, #dB>, %acc: tensor<16x8xf32, #mma>)
      -> tensor<16x8xf32, #mma> {
    %r = tt.reshape %w : tensor<16x32xf16, #blocked> -> tensor<16x2x16xf16, #b3>
    %tr = tt.trans %r {order = array<i32: 0, 2, 1>} : tensor<16x2x16xf16, #b3> -> tensor<16x16x2xf16, #t3>
    %s0, %s1 = tt.split %tr : tensor<16x16x2xf16, #t3> -> tensor<16x16xf16, #blocked>
    %c0 = ttg.convert_layout %s0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #dA>
    %c1 = ttg.convert_layout %s1 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #dA>
    %d0 = tt.dot %c0, %r0, %acc : tensor<16x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<16x8xf32, #mma>
    %d1 = tt.dot %c1, %r1, %d0 : tensor<16x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<16x8xf32, #mma>
    tt.return %d1 : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// An extract reads `[index * itsOwnExtent, +itsOwnExtent)`, so consecutive
// indices only tile the source when the extents agree. These three read
// `w[0:32] w[16:32] w[32:48]` -- one slice twice, one never -- yet they
// reassemble to `w`'s type, so the fold has to reject them on the offsets
// rather than on the type alone.
// MERGE-LABEL: @keep_concat_of_overlapping_extracts
// MERGE: ttg.extract_dot_operand
// MERGE: ttg.concat_dot_operand
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @keep_concat_of_overlapping_extracts(%w: tensor<16x64xf16, #dA>, %rhs: tensor<64x8xf16, #dB>,
                                               %acc: tensor<16x8xf32, #mma>) -> tensor<16x8xf32, #mma> {
    %e0 = ttg.extract_dot_operand %w {dim = 1 : i32, index = 0 : i32} : tensor<16x64xf16, #dA> -> tensor<16x32xf16, #dA>
    %e1 = ttg.extract_dot_operand %w {dim = 1 : i32, index = 1 : i32} : tensor<16x64xf16, #dA> -> tensor<16x16xf16, #dA>
    %e2 = ttg.extract_dot_operand %w {dim = 1 : i32, index = 2 : i32} : tensor<16x64xf16, #dA> -> tensor<16x16xf16, #dA>
    %c = ttg.concat_dot_operand %e0, %e1, %e2 {dim = 1 : i32} : tensor<16x32xf16, #dA>, tensor<16x16xf16, #dA>, tensor<16x16xf16, #dA> -> tensor<16x64xf16, #dA>
    %d = tt.dot %c, %rhs, %acc : tensor<16x64xf16, #dA> * tensor<64x8xf16, #dB> -> tensor<16x8xf32, #mma>
    tt.return %d : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#bj = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 8, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#bt = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 1, 8], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>
#blin = #ttg.linear<{register = [[1, 0]], lane = [[0, 1], [0, 2], [0, 4], [2, 0], [4, 0]], warp = [], block = []}>

// Interleaved: the permutation moves the fragment axis *after* the in-fragment
// K coordinate -- `permute(0, 2, 1)` on a (KP, N, 2) join -- so result element
// `2j + i` is element `j` of fragment `i`. This is the shape plain sub-byte
// packing produces; no separate nibble-select is needed. The flag is in the op
// because the register map differs from the concatenating form, and the expand
// pass emits the matching `permute(0, 2, 1)` rather than the usual one.
// MATCH-LABEL: @interleave_b_side
// MATCH-NOT: tt.join
// MATCH: ttg.concat_dot_operand %arg0, %arg1 {dim = 0 : i32, interleaved}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @interleave_b_side(%lo: tensor<4x8xf16, #blocked>, %hi: tensor<4x8xf16, #blocked>,
                              %lhs: tensor<8x8xf16, #dA>, %acc: tensor<8x8xf32, #mma>) -> tensor<8x8xf32, #mma> {
    %j = tt.join %lo, %hi : tensor<4x8xf16, #blocked> -> tensor<4x8x2xf16, #bj>
    %tr = tt.trans %j {order = array<i32: 0, 2, 1>} : tensor<4x8x2xf16, #bj> -> tensor<4x2x8xf16, #bt>
    %r = tt.reshape %tr : tensor<4x2x8xf16, #bt> -> tensor<8x8xf16, #blin>
    %c = ttg.convert_layout %r : tensor<8x8xf16, #blin> -> tensor<8x8xf16, #dB>
    %d = tt.dot %lhs, %c, %acc : tensor<8x8xf16, #dA> * tensor<8x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d : tensor<8x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#bj = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 8, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#bt = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 1, 8], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>
#blin = #ttg.linear<{register = [[1, 0]], lane = [[0, 1], [0, 2], [0, 4], [2, 0], [4, 0]], warp = [], block = []}>

// Expand puts the interleaved permutation back. The result must look like the
// input to the matcher: join -> trans(0,2,1) -> reshape.
// ROUNDTRIP-LABEL: @expand_interleaved_concat
// ROUNDTRIP: tt.join
// ROUNDTRIP: tt.trans {{.*order = array<i32: 0, 2, 1>}}
// ROUNDTRIP: tt.reshape
// ROUNDTRIP-NOT: ttg.concat_dot_operand
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @expand_interleaved_concat(%lo: tensor<4x8xf16, #blocked>, %hi: tensor<4x8xf16, #blocked>,
                                     %lhs: tensor<8x8xf16, #dA>, %acc: tensor<8x8xf32, #mma>) -> tensor<8x8xf32, #mma> {
    %j = tt.join %lo, %hi : tensor<4x8xf16, #blocked> -> tensor<4x8x2xf16, #bj>
    %tr = tt.trans %j {order = array<i32: 0, 2, 1>} : tensor<4x8x2xf16, #bj> -> tensor<4x2x8xf16, #bt>
    %r = tt.reshape %tr : tensor<4x2x8xf16, #bt> -> tensor<8x8xf16, #blin>
    %c = ttg.convert_layout %r : tensor<8x8xf16, #blin> -> tensor<8x8xf16, #dB>
    %d = tt.dot %lhs, %c, %acc : tensor<8x8xf16, #dA> * tensor<8x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d : tensor<8x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// Interleaved fragments that already carry a dot_op encoding, which is what
// reaches the interleaved arithmetic in the register map: everything the map
// checks before that point passes here. It still has to expand, because a
// dot_op thread owns its K elements contiguously and alternating fragments
// every element puts the data it needs on other threads. Running the
// concatenating arithmetic over this op instead would report a relabel that
// silently reads the wrong elements.
// ROUNDTRIP-LABEL: @interleave_dot_op_fragments
// ROUNDTRIP-NOT: ttg.concat_dot_operand
// ROUNDTRIP: tt.join
// ROUNDTRIP: tt.reshape
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @interleave_dot_op_fragments(%a0: tensor<16x16xf16, #dA>, %a1: tensor<16x16xf16, #dA>,
                                       %rhs: tensor<32x8xf16, #dB>, %acc: tensor<16x8xf32, #mma>) -> tensor<16x8xf32, #mma> {
    %w = ttg.concat_dot_operand %a0, %a1 {dim = 1 : i32, interleaved} : tensor<16x16xf16, #dA>, tensor<16x16xf16, #dA> -> tensor<16x32xf16, #dA>
    %d = tt.dot %w, %rhs, %acc : tensor<16x32xf16, #dA> * tensor<32x8xf16, #dB> -> tensor<16x8xf32, #mma>
    tt.return %d : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>

// Nested interleaving is not flat interleaving: stacking pairs and then
// stacking the results yields a0 c0 b0 d0, while one four-way interleave yields
// a0 b0 c0 d0. Both levels must survive.
// MATCH-LABEL: @keep_nested_interleave
// MATCH-COUNT-3: ttg.concat_dot_operand
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @keep_nested_interleave(%a0: tensor<8x4xf16, #blocked>, %a1: tensor<8x4xf16, #blocked>,
                                  %a2: tensor<8x4xf16, #blocked>, %a3: tensor<8x4xf16, #blocked>,
                                  %rhs: tensor<16x8xf16, #dB>, %acc: tensor<8x8xf32, #mma>) -> tensor<8x8xf32, #mma> {
    %s0 = ttg.concat_dot_operand %a0, %a1 {dim = 1 : i32, interleaved} : tensor<8x4xf16, #blocked>, tensor<8x4xf16, #blocked> -> tensor<8x8xf16, #blocked>
    %s1 = ttg.concat_dot_operand %a2, %a3 {dim = 1 : i32, interleaved} : tensor<8x4xf16, #blocked>, tensor<8x4xf16, #blocked> -> tensor<8x8xf16, #blocked>
    %w = ttg.concat_dot_operand %s0, %s1 {dim = 1 : i32, interleaved} : tensor<8x8xf16, #blocked>, tensor<8x8xf16, #blocked> -> tensor<8x16xf16, #blocked>
    %c = ttg.convert_layout %w : tensor<8x16xf16, #blocked> -> tensor<8x16xf16, #dA>
    %d = tt.dot %c, %rhs, %acc : tensor<8x16xf16, #dA> * tensor<16x8xf16, #dB> -> tensor<8x8xf32, #mma>
    tt.return %d : tensor<8x8xf32, #mma>
  }
}
