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

// RUN: triton-opt %s -triton-tle-lower-barriers -split-input-file | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @lower_mbarrier
  tt.func @lower_mbarrier() {
    %c0 = arith.constant 0 : i32
    %bars = tle.barrier.alloc {arrive_count = 2 : i32, init_polarity = 0 : i32, num_barriers = 2 : i32} : !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    %slot = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %slot, %c0 {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.arrive %slot, %c0 {arrive_count = 2 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// CHECK: %[[ALLOC:.*]] = ttg.local_alloc
// CHECK: ttng.init_barrier
// CHECK: ttng.init_barrier
// CHECK: ttng.wait_barrier
// CHECK: ttng.arrive_barrier
// CHECK-NOT: tle.barrier

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @lower_ready_mbarrier_no_initial_arrive
  tt.func @lower_ready_mbarrier_no_initial_arrive() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %bars = tle.barrier.alloc {arrive_count = 2 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %slot = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %slot, %c1 {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// CHECK: ttng.init_barrier
// CHECK-NOT: ttng.arrive_barrier
// CHECK: ttng.wait_barrier
// CHECK-NOT: tle.barrier

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @lower_named_barrier
  tt.func @lower_named_barrier(%slot: !ttg.memdesc<1xi64, #slot, #smem, mutable>) {
    tle.barrier.wait %slot {backend = "named", named_id = 3 : i32, named_num_threads = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.arrive %slot {arrive_count = 1 : i32, backend = "named", named_id = 3 : i32, named_num_threads = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// CHECK: %[[ID:.*]] = arith.constant 3 : i32
// CHECK: %[[THREADS:.*]] = arith.constant 256 : i32
// CHECK: ttng.wait_barrier_named %[[ID]], %[[THREADS]]
// CHECK: ttng.arrive_barrier_named
// CHECK-NOT: tle.barrier
