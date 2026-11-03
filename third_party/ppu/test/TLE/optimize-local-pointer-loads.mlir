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

// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-loads=keep-smem-read-order=true | FileCheck %s --check-prefixes=COMMON,ON
// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-loads | FileCheck %s --check-prefixes=COMMON,OFF

// With the option (FLAGTREE_KEEP_SMEM_READ_ORDER), a local pointer load is
// re-issued at the layout conversion that uses it only when nothing between
// the two writes shared memory: the load would read the rewritten data
// otherwise. Without it the load is re-issued regardless.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: tt.func @rematerialize_without_write_in_between
  tt.func @rematerialize_without_write_in_between(%slot_scalar: i32, %v: tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked_alt> {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // COMMON-NOT: ttg.convert_layout
    %out = ttg.convert_layout %ids : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#blocked_alt = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [2], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 2 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: tt.func @keep_convert_after_write
  tt.func @keep_convert_after_write(%slot_scalar: i32, %v: tensor<64xi32, #blocked>) -> tensor<64xi32, #blocked_alt> {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2x64xi32, #shared, #smem, mutable>
    %slot = tt.splat %slot_scalar : i32 -> tensor<64xi32, #blocked>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%smem, %slot, %offs) : (!ttg.memdesc<2x64xi32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<i32, 3>, #blocked>
    // ON: %[[IDS:.*]] = tt.load
    // OFF-NOT: tt.load
    %ids = tt.load %ptr : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // ON-NEXT: tt.store
    // OFF: tt.store
    tt.store %ptr, %v : tensor<64x!tt.ptr<i32, 3>, #blocked>
    // The conversion follows the store: re-issuing the load there would read the new data.
    // ON: ttg.convert_layout %[[IDS]]
    // OFF: tt.load
    // OFF-NOT: ttg.convert_layout
    %out = ttg.convert_layout %ids : tensor<64xi32, #blocked> -> tensor<64xi32, #blocked_alt>
    tt.return %out : tensor<64xi32, #blocked_alt>
  }
}
