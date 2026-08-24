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
// RUN: triton-opt %s -split-input-file -tritongpu-merge-segmented-dot -canonicalize | FileCheck %s --check-prefix=MERGE

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#dA = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#dB = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>
#j = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 4, 1], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#t = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [8, 1, 4], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>
#lin = #ttg.linear<{register = [[0, 4]], lane = [[0, 1], [0, 2], [1, 0], [2, 0], [4, 0]], warp = [], block = []}>

// A two-leaf tree whose result reaches a dot: join -> trans -> reshape is an
// ordered concat along K, so fold it into one op. The convert in between is the
// state the pass sees, since it runs before the operand layouts are assigned.
// MATCH-LABEL: @concat_two
// MATCH-NOT: tt.join
// MATCH: ttg.concat_dot_operand %arg0, %arg1 {dim = 1 : i32}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @concat_two(%a: tensor<8x4xf16, #blocked>, %b: tensor<8x4xf16, #blocked>,
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

// Folding a three-level tree and expanding it again must restore the same chain:
// the operands here keep a blocked layout, which the relabel cannot use, so the
// expansion runs and has to be an exact inverse of the match.
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

// The pass folds a segmented K chain back into one dot when, and only when, the
// operands are provably a complete cover of one wide value and merging cannot
// extend a live range. Cases below pair each accepted shape with the negative
// that must stay segmented.

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A complete cover folds straight back to the wide operands, so no concat
  // survives and no extract is left behind.
  // MERGE-LABEL: @merge_extract_cover
  // MERGE-NOT: ttg.extract_dot_operand
  // MERGE-COUNT-1: tt.dot
  // MERGE-NOT: tt.dot
  tt.func @merge_extract_cover(
      %aw: tensor<16x32xf16, #a>, %bw: tensor<32x8xf16, #b>,
      %acc: tensor<16x8xf32, #mma>) -> tensor<16x8xf32, #mma> {
    %a0 = ttg.extract_dot_operand %aw {dim = 1 : i32, index = 0 : i32} : tensor<16x32xf16, #a> -> tensor<16x16xf16, #a>
    %a1 = ttg.extract_dot_operand %aw {dim = 1 : i32, index = 1 : i32} : tensor<16x32xf16, #a> -> tensor<16x16xf16, #a>
    %b0 = ttg.extract_dot_operand %bw {dim = 0 : i32, index = 0 : i32} : tensor<32x8xf16, #b> -> tensor<16x8xf16, #b>
    %b1 = ttg.extract_dot_operand %bw {dim = 0 : i32, index = 1 : i32} : tensor<32x8xf16, #b> -> tensor<16x8xf16, #b>
    %d0 = tt.dot %a0, %b0, %acc : tensor<16x16xf16, #a> * tensor<16x8xf16, #b> -> tensor<16x8xf32, #mma>
    %d1 = tt.dot %a1, %b1, %d0 : tensor<16x16xf16, #a> * tensor<16x8xf16, #b> -> tensor<16x8xf32, #mma>
    tt.return %d1 : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

// What the pipeline really produces: the split-tree matcher wraps one wide
// value in a separate convert per extract, and the pass runs before layout
// cleanup so the accumulator hops through a convert pair between the dots.
// Both must be looked through, or a cover that is complete by construction
// reads as unrelated fragments and the chain looks broken at its first link.
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // MERGE-LABEL: @merge_through_converts
  // MERGE-COUNT-1: tt.dot
  // MERGE-NOT: tt.dot
  tt.func @merge_through_converts(
      %aw: tensor<16x32xf16, #blocked>, %bw: tensor<32x8xf16, #blocked>,
      %acc: tensor<16x8xf16, #mma>) -> tensor<16x8xf16, #mma> {
    %aw0 = ttg.convert_layout %aw : tensor<16x32xf16, #blocked> -> tensor<16x32xf16, #a>
    %aw1 = ttg.convert_layout %aw : tensor<16x32xf16, #blocked> -> tensor<16x32xf16, #a>
    %bw0 = ttg.convert_layout %bw : tensor<32x8xf16, #blocked> -> tensor<32x8xf16, #b>
    %bw1 = ttg.convert_layout %bw : tensor<32x8xf16, #blocked> -> tensor<32x8xf16, #b>
    %a0 = ttg.extract_dot_operand %aw0 {dim = 1 : i32, index = 0 : i32} : tensor<16x32xf16, #a> -> tensor<16x16xf16, #a>
    %a1 = ttg.extract_dot_operand %aw1 {dim = 1 : i32, index = 1 : i32} : tensor<16x32xf16, #a> -> tensor<16x16xf16, #a>
    %b0 = ttg.extract_dot_operand %bw0 {dim = 0 : i32, index = 0 : i32} : tensor<32x8xf16, #b> -> tensor<16x8xf16, #b>
    %b1 = ttg.extract_dot_operand %bw1 {dim = 0 : i32, index = 1 : i32} : tensor<32x8xf16, #b> -> tensor<16x8xf16, #b>
    %d0 = tt.dot %a0, %b0, %acc : tensor<16x16xf16, #a> * tensor<16x8xf16, #b> -> tensor<16x8xf16, #mma>
    %r0 = ttg.convert_layout %d0 : tensor<16x8xf16, #mma> -> tensor<16x8xf16, #blocked>
    %r1 = ttg.convert_layout %r0 : tensor<16x8xf16, #blocked> -> tensor<16x8xf16, #mma>
    %d1 = tt.dot %a1, %b1, %r1 : tensor<16x16xf16, #a> * tensor<16x8xf16, #b> -> tensor<16x8xf16, #mma>
    tt.return %d1 : tensor<16x8xf16, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>

// The converted partial sum escapes the chain, so it stays observable and must
// keep being computed: looking through accumulator converts may not drop the
// single-use requirement.
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // MERGE-LABEL: @keep_escaping_partial_sum
  // MERGE-COUNT-2: tt.dot
  tt.func @keep_escaping_partial_sum(
      %aw: tensor<16x32xf16, #a>, %bw: tensor<32x8xf16, #b>,
      %acc: tensor<16x8xf16, #mma>)
      -> (tensor<16x8xf16, #mma>, tensor<16x8xf16, #blocked>) {
    %a0 = ttg.extract_dot_operand %aw {dim = 1 : i32, index = 0 : i32} : tensor<16x32xf16, #a> -> tensor<16x16xf16, #a>
    %a1 = ttg.extract_dot_operand %aw {dim = 1 : i32, index = 1 : i32} : tensor<16x32xf16, #a> -> tensor<16x16xf16, #a>
    %b0 = ttg.extract_dot_operand %bw {dim = 0 : i32, index = 0 : i32} : tensor<32x8xf16, #b> -> tensor<16x8xf16, #b>
    %b1 = ttg.extract_dot_operand %bw {dim = 0 : i32, index = 1 : i32} : tensor<32x8xf16, #b> -> tensor<16x8xf16, #b>
    %d0 = tt.dot %a0, %b0, %acc : tensor<16x16xf16, #a> * tensor<16x8xf16, #b> -> tensor<16x8xf16, #mma>
    %r0 = ttg.convert_layout %d0 : tensor<16x8xf16, #mma> -> tensor<16x8xf16, #blocked>
    %r1 = ttg.convert_layout %r0 : tensor<16x8xf16, #blocked> -> tensor<16x8xf16, #mma>
    %d1 = tt.dot %a1, %b1, %r1 : tensor<16x16xf16, #a> * tensor<16x8xf16, #b> -> tensor<16x8xf16, #mma>
    tt.return %d1, %r0 : tensor<16x8xf16, #mma>, tensor<16x8xf16, #blocked>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

// Independent operands are not a cover of one wide value, so gathering them
// would extend live ranges rather than shorten them.
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // MERGE-LABEL: @keep_independent_fragments
  // MERGE-NOT: ttg.concat_dot_operand
  // MERGE-COUNT-2: tt.dot
  tt.func @keep_independent_fragments(
      %a0: tensor<16x16xf16, #a>, %a1: tensor<16x16xf16, #a>,
      %b0: tensor<16x8xf16, #b>, %b1: tensor<16x8xf16, #b>,
      %acc: tensor<16x8xf32, #mma>) -> tensor<16x8xf32, #mma> {
    %d0 = tt.dot %a0, %b0, %acc : tensor<16x16xf16, #a> * tensor<16x8xf16, #b> -> tensor<16x8xf32, #mma>
    %d1 = tt.dot %a1, %b1, %d0 : tensor<16x16xf16, #a> * tensor<16x8xf16, #b> -> tensor<16x8xf32, #mma>
    tt.return %d1 : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#parent = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [1, 1], order = [1, 0]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
// Layout propagation leaves transient dot_op encodings whose parent is still a
// blocked layout. No mma reads those, so the extracts must be built against the
// encoding the dot actually consumes.
#transient = #ttg.dot_op<{opIdx = 0, parent = #parent}>
#f3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 1, 8], warpsPerCTA = [1, 1, 1], order = [2, 1, 0]}>
#t3 = #ttg.blocked<{sizePerThread = [1, 1, 1], threadsPerWarp = [4, 8, 1], warpsPerCTA = [1, 1, 1], order = [1, 2, 0]}>

module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // Extracting against #transient would build a cover the concat cannot fold
  // back, leaving fragments for ReduceDataDuplication to stage through shared
  // memory.
  // Extracting against the mma operand layout makes the rebuilt wide value fold
  // all the way back, so the split tree, the extracts and the concat all die and
  // one wide dot is left. Extracting against #transient instead would leave the
  // fragments behind for ReduceDataDuplication to stage through shared memory.
  // MERGE-LABEL: @split_tree_extracts_use_the_mma_operand_layout
  // MERGE-NOT: tt.split
  // MERGE-NOT: ttg.extract_dot_operand
  // MERGE-NOT: ttg.concat_dot_operand
  // MERGE: tt.dot
  // MERGE-SAME: tensor<16x32xf16
  // MERGE-NOT: tt.dot
  tt.func @split_tree_extracts_use_the_mma_operand_layout(
      %aw: tensor<16x32xf16, #blocked>, %bw: tensor<32x8xf16, #b>,
      %acc: tensor<16x8xf32, #mma>) -> tensor<16x8xf32, #mma> {
    %r = tt.reshape %aw : tensor<16x32xf16, #blocked> -> tensor<16x2x16xf16, #f3>
    %t = tt.trans %r {order = array<i32: 0, 2, 1>} : tensor<16x2x16xf16, #f3> -> tensor<16x16x2xf16, #t3>
    %f0, %f1 = tt.split %t : tensor<16x16x2xf16, #t3> -> tensor<16x16xf16, #blocked>
    %b0 = ttg.extract_dot_operand %bw {dim = 0 : i32, index = 0 : i32} : tensor<32x8xf16, #b> -> tensor<16x8xf16, #b>
    %b1 = ttg.extract_dot_operand %bw {dim = 0 : i32, index = 1 : i32} : tensor<32x8xf16, #b> -> tensor<16x8xf16, #b>
    // Each leaf reaches its dot through a transient dot_op convert first.
    %p0 = ttg.convert_layout %f0 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #transient>
    %o0 = ttg.convert_layout %p0 : tensor<16x16xf16, #transient> -> tensor<16x16xf16, #a>
    %d0 = tt.dot %o0, %b0, %acc : tensor<16x16xf16, #a> * tensor<16x8xf16, #b> -> tensor<16x8xf32, #mma>
    %p1 = ttg.convert_layout %f1 : tensor<16x16xf16, #blocked> -> tensor<16x16xf16, #transient>
    %o1 = ttg.convert_layout %p1 : tensor<16x16xf16, #transient> -> tensor<16x16xf16, #a>
    %d1 = tt.dot %o1, %b1, %d0 : tensor<16x16xf16, #a> * tensor<16x8xf16, #b> -> tensor<16x8xf32, #mma>
    tt.return %d1 : tensor<16x8xf32, #mma>
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 1], instrShape = [16, 8]}>
#a = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>

// The identity fold is a canonicalization on the op, not a pattern private to
// the merge pass: a chain's extracts only become recognizable as a cover of one
// root after layout propagation unifies their sources, which happens after that
// pass has run.
module attributes {"ttg.target" = "cuda:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // MERGE-LABEL: @canonicalize_concat_of_complete_cover
  // MERGE-NOT: ttg.concat_dot_operand
  // MERGE-NOT: ttg.extract_dot_operand
  // MERGE: tt.return %arg0
  tt.func @canonicalize_concat_of_complete_cover(
      %aw: tensor<16x32xf16, #a>) -> tensor<16x32xf16, #a> {
    %a0 = ttg.extract_dot_operand %aw {dim = 1 : i32, index = 0 : i32} : tensor<16x32xf16, #a> -> tensor<16x16xf16, #a>
    %a1 = ttg.extract_dot_operand %aw {dim = 1 : i32, index = 1 : i32} : tensor<16x32xf16, #a> -> tensor<16x16xf16, #a>
    %w = ttg.concat_dot_operand %a0, %a1 {dim = 1 : i32} : tensor<16x16xf16, #a>, tensor<16x16xf16, #a> -> tensor<16x32xf16, #a>
    tt.return %w : tensor<16x32xf16, #a>
  }
}
