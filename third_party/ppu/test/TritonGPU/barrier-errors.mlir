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

// RUN: triton-opt %s -split-input-file --allocate-shared-memory-ppu --convert-triton-gpu-to-llvm-ppu -verify-diagnostics

// Barrier forms the PPU LLVM conversion rejects, each with the reason before
// the generic legalization failure. The PPU compiler rejects the TLE API that
// produces the mbarrier ops on the frontend IR (compiler.reject_unsupported_tle);
// these are the backstop for IR that reaches the conversion another way.

// A named barrier whose thread count is not a constant: the SDK builtin takes
// it as an immediate.

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @named_wait_dynamic_count(%threads: i32) {
    %id = arith.constant 3 : i32
    // expected-error @below {{PPU named barriers need a constant thread count}}
    // expected-error @below {{failed to legalize operation 'ttng.wait_barrier_named' that was explicitly marked illegal}}
    ttng.wait_barrier_named %id, %threads : i32, i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @named_arrive_dynamic_count(%threads: i32) {
    %id = arith.constant 3 : i32
    // expected-error @below {{PPU named barriers need a constant thread count}}
    // expected-error @below {{failed to legalize operation 'ttng.arrive_barrier_named' that was explicitly marked illegal}}
    ttng.arrive_barrier_named %id, %threads : i32, i32
    tt.return
  }
}

// -----

// The mbarrier ops: what TleLowerBarriers emits for a barrier that is not
// named-only (init_barrier), the phased wait and arrive, and the transaction
// forms a descriptor copy with a completion barrier needs.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mbarrier_init() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    // expected-error @below {{the PPU backend has no mbarrier; use named barriers (tle.gpu.barrier_wait/barrier_arrive without phaseIdx)}}
    // expected-error @below {{failed to legalize operation 'ttng.init_barrier' that was explicitly marked illegal}}
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mbarrier_wait(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    %phase = arith.constant 0 : i32
    // expected-error @below {{the PPU backend has no mbarrier; use named barriers (tle.gpu.barrier_wait/barrier_arrive without phaseIdx)}}
    // expected-error @below {{failed to legalize operation 'ttng.wait_barrier' that was explicitly marked illegal}}
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mbarrier_arrive(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @below {{the PPU backend has no mbarrier; use named barriers (tle.gpu.barrier_wait/barrier_arrive without phaseIdx)}}
    // expected-error @below {{failed to legalize operation 'ttng.arrive_barrier' that was explicitly marked illegal}}
    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mbarrier_expect(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    %true = arith.constant true
    // expected-error @below {{the PPU backend has no mbarrier; use named barriers (tle.gpu.barrier_wait/barrier_arrive without phaseIdx)}}
    // expected-error @below {{failed to legalize operation 'ttng.barrier_expect' that was explicitly marked illegal}}
    ttng.barrier_expect %bar, 4096, %true : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mbarrier_inval(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @below {{the PPU backend has no mbarrier; use named barriers (tle.gpu.barrier_wait/barrier_arrive without phaseIdx)}}
    // expected-error @below {{failed to legalize operation 'ttng.inval_barrier' that was explicitly marked illegal}}
    ttng.inval_barrier %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @mbarrier_async_copy_arrive(%bar: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // expected-error @below {{the PPU backend has no mbarrier; use named barriers (tle.gpu.barrier_wait/barrier_arrive without phaseIdx)}}
    // expected-error @below {{failed to legalize operation 'ttng.async_copy_mbarrier_arrive' that was explicitly marked illegal}}
    ttng.async_copy_mbarrier_arrive %bar : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
