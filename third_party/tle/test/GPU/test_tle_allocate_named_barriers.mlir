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

// RUN: triton-opt %s -triton-tle-allocate-named-barriers -split-input-file | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @remap_virtual_ids_without_warpspec
  tt.func @remap_virtual_ids_without_warpspec() {
    %v0 = arith.constant 16 : i32
    %v1 = arith.constant 17 : i32
    %threads = arith.constant 256 : i32

    // CHECK: %[[ID0:.+]] = arith.constant 1 : i32
    // CHECK: ttng.wait_barrier_named %[[ID0]], {{.*}} : i32, i32
    ttng.wait_barrier_named %v0, %threads : i32, i32

    // CHECK: %[[ID1:.+]] = arith.constant 2 : i32
    // CHECK: ttng.arrive_barrier_named %[[ID1]], {{.*}} : i32, i32
    ttng.arrive_barrier_named %v1, %threads : i32, i32

    // CHECK: %[[ID0_AGAIN:.+]] = arith.constant 1 : i32
    // CHECK: ttng.arrive_barrier_named %[[ID0_AGAIN]], {{.*}} : i32, i32
    ttng.arrive_barrier_named %v0, %threads : i32, i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @remap_virtual_ids_with_warpspec
  tt.func @remap_virtual_ids_with_warpspec() {
    ttg.warp_specialize() attributes {requestedRegisters = array<i32: 128, 128>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %v0 = arith.constant 16 : i32
      %threads = arith.constant 256 : i32
      // CHECK: %[[ID4:.+]] = arith.constant 4 : i32
      // CHECK: ttng.wait_barrier_named %[[ID4]], {{.*}} : i32, i32
      ttng.wait_barrier_named %v0, %threads : i32, i32
      ttg.warp_return
    }
    partition1() num_warps(4) {
      %v1 = arith.constant 17 : i32
      %threads = arith.constant 256 : i32
      // CHECK: %[[ID5:.+]] = arith.constant 5 : i32
      // CHECK: ttng.arrive_barrier_named %[[ID5]], {{.*}} : i32, i32
      ttng.arrive_barrier_named %v1, %threads : i32, i32
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @preserve_existing_physical_ids_with_warpspec
  tt.func @preserve_existing_physical_ids_with_warpspec() {
    ttg.warp_specialize() attributes {requestedRegisters = array<i32: 128, 128>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %physical0 = arith.constant 0 : i32
      %threads = arith.constant 256 : i32
      // CHECK: %[[PHYSICAL0:.+]] = arith.constant 0 : i32
      // CHECK: ttng.wait_barrier_named %[[PHYSICAL0]], {{.*}} : i32, i32
      ttng.wait_barrier_named %physical0, %threads : i32, i32
      ttg.warp_return
    }
    partition1() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

// Reused single-warp partitions use warp sync, so reserve only IDs 0 and 1.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @remap_virtual_id_after_sixteen_reused_warps
  tt.func @remap_virtual_id_after_sixteen_reused_warps() {
    ttg.warp_specialize() attributes {reuseDefaultWarps = true}
    default { ttg.warp_yield }
    partition0() num_warps(1) { ttg.warp_return }
    partition1() num_warps(1) { ttg.warp_return }
    partition2() num_warps(1) { ttg.warp_return }
    partition3() num_warps(1) { ttg.warp_return }
    partition4() num_warps(1) { ttg.warp_return }
    partition5() num_warps(1) { ttg.warp_return }
    partition6() num_warps(1) { ttg.warp_return }
    partition7() num_warps(1) { ttg.warp_return }
    partition8() num_warps(1) { ttg.warp_return }
    partition9() num_warps(1) { ttg.warp_return }
    partition10() num_warps(1) { ttg.warp_return }
    partition11() num_warps(1) { ttg.warp_return }
    partition12() num_warps(1) { ttg.warp_return }
    partition13() num_warps(1) { ttg.warp_return }
    partition14() num_warps(1) { ttg.warp_return }
    partition15() num_warps(1) { ttg.warp_return } : () -> ()
    %virtual = arith.constant 16 : i32
    %threads = arith.constant 512 : i32
    // CHECK: %[[ID2:.+]] = arith.constant 2 : i32
    // CHECK: ttng.wait_barrier_named %[[ID2]], {{.*}} : i32, i32
    ttng.wait_barrier_named %virtual, %threads : i32, i32
    tt.return
  }
}

// -----

// Reserve the multi-warp partition's ID even before LLVM inserts barriers.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @reserve_reused_multi_warp_barrier
  tt.func @reserve_reused_multi_warp_barrier() {
    ttg.warp_specialize() attributes {reuseDefaultWarps = true}
    default { ttg.warp_yield }
    partition0() num_warps(1) { ttg.warp_return }
    partition1() num_warps(2) { ttg.warp_return }
    partition2() num_warps(1) { ttg.warp_return } : () -> ()
    %virtual = arith.constant 16 : i32
    %threads = arith.constant 128 : i32
    // CHECK: %[[ID3:.+]] = arith.constant 3 : i32
    // CHECK: ttng.wait_barrier_named %[[ID3]], {{.*}} : i32, i32
    ttng.wait_barrier_named %virtual, %threads : i32, i32
    tt.return
  }
}
