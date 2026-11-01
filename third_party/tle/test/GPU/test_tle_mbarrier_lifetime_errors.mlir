// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -triton-tle-lower-barriers -split-input-file -verify-diagnostics

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module {
  tt.func @return_owned_barrier() -> !ttg.memdesc<1x1xi64, #shared, #smem, mutable> {
    // expected-error @+1 {{owned mbarrier cannot escape its defining region}}
    %bar = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    tt.return %bar : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module {
  tt.func @return_owned_slot() -> !ttg.memdesc<1xi64, #slot, #smem, mutable> {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{owned mbarrier cannot escape its defining region}}
    %bar = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %bar[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return %s : !ttg.memdesc<1xi64, #slot, #smem, mutable>
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module {
  tt.func @yield_owned_barrier(%condition: i1) {
    %result = scf.if %condition -> (!ttg.memdesc<1x1xi64, #shared, #smem, mutable>) {
      // expected-error @+1 {{owned mbarrier cannot escape its defining region}}
      %bar = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
      scf.yield %bar : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    } else {
      // expected-error @+1 {{owned mbarrier cannot escape its defining region}}
      %other = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
      scf.yield %other : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    }
    tt.return
  }
}
