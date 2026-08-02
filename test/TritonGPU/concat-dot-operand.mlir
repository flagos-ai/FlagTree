// RUN: triton-opt %s -split-input-file -tritongpu-concat-dot-operand | FileCheck %s --check-prefixes=COMMON,MATCH
// RUN: triton-opt %s -split-input-file -tritongpu-concat-dot-operand -tritongpu-expand-concat-dot-operand | FileCheck %s --check-prefixes=COMMON,ROUNDTRIP

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
