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

// RUN: triton-opt %s -split-input-file -triton-tle-schedule-tma-store-sync -allocate-shared-memory | FileCheck %s

// The wait names the source it completes, so the source stays live until the
// wait and shared-memory allocation cannot place a later buffer, or the
// scratch of an op such as convert_layout, at its offset while the TMA engine
// may still read it.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @convert_layout_scratch_does_not_reuse_source
  tt.func public @convert_layout_scratch_does_not_reuse_source(%desc: !tt.tensordesc<tensor<32x32xf32, #shared>>, %v: tensor<32x32xf32, #blocked>, %x: tensor<32x32xf32, #blocked>) -> tensor<32x32xf32, #blocked_t> {
    %c0 = arith.constant 0 : i32
    // CHECK: %[[SRC:.+]] = ttg.local_alloc %{{.+}} {allocation.offset = 0 : i32}
    %src = ttg.local_alloc %v : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global %{{.+}} %[[SRC]]
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src {tle.tma_store_explicit_commit} : !tt.tensordesc<tensor<32x32xf32, #shared>>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    tle.tma_store.commit_group
    ttng.async_tma_store_wait {pendings = 0 : i32}
    // CHECK-NEXT: tle.tma_store.commit_group
    // The source is 4096 bytes at offset 0; the scratch must not overlap it.
    // CHECK-NEXT: ttg.convert_layout %{{.+}} {allocation.offset = 4096 : i32}
    %y = ttg.convert_layout %x : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_t>
    // CHECK-NEXT: ttng.async_tma_store_wait %[[SRC]] : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> {pendings = 0 : i32}
    // CHECK-NEXT: tt.return
    tt.return %y : tensor<32x32xf32, #blocked_t>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked_t = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // A group may stay pending across a region exit when its source outlives
  // the region. The wait after the region names the source, so it stays live
  // across the conversion in between and the scratch cannot take its offset.
  // CHECK-LABEL: @source_outside_region_live_until_wait
  tt.func public @source_outside_region_live_until_wait(%desc: !tt.tensordesc<tensor<32x32xf32, #shared>>, %v: tensor<32x32xf32, #blocked>, %x: tensor<32x32xf32, #blocked>, %cond: i1) -> tensor<32x32xf32, #blocked_t> {
    %c0 = arith.constant 0 : i32
    // CHECK: %[[SRC:.+]] = ttg.local_alloc %{{.+}} {allocation.offset = 0 : i32}
    %src = ttg.local_alloc %v : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    // CHECK-NEXT: scf.if
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global %{{.+}} %[[SRC]]
    // CHECK-NEXT: tle.tma_store.commit_group
    // CHECK-NEXT: }
    scf.if %cond {
      ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src {tle.tma_store_explicit_commit} : !tt.tensordesc<tensor<32x32xf32, #shared>>, !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
      tle.tma_store.commit_group
      ttng.async_tma_store_wait {pendings = 0 : i32}
      scf.yield
    }
    // CHECK-NEXT: ttg.convert_layout %{{.+}} {allocation.offset = 4096 : i32}
    %y = ttg.convert_layout %x : tensor<32x32xf32, #blocked> -> tensor<32x32xf32, #blocked_t>
    // CHECK-NEXT: ttng.async_tma_store_wait %[[SRC]] : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> {pendings = 0 : i32}
    // CHECK-NEXT: tt.return
    tt.return %y : tensor<32x32xf32, #blocked_t>
  }
}
