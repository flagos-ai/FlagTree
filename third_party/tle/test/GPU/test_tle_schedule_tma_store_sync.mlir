/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
* Copyright 2025-     FlagOS Contributors
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// RUN: triton-opt %s -split-input-file -triton-tle-schedule-tma-store-sync | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @sink_wait_to_shared_reuse
  tt.func public @sink_wait_to_shared_reuse(%desc: !tt.tensordesc<tensor<16x16xf32, #shared>>, %src: !ttg.memdesc<16x16xf32, #shared, #smem, mutable>, %dst: !ttg.memdesc<16x16xf32, #shared, #smem, mutable>, %x: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    // CHECK: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src {tle.tma_store_explicit_commit} : !tt.tensordesc<tensor<16x16xf32, #shared>>, !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    tle.tma_store.commit_group
    ttng.async_tma_store_wait {pendings = 0 : i32}
    // CHECK-NEXT: tle.tma_store.commit_group
    // CHECK-NEXT: arith.addi
    %y = arith.addi %x, %c1 : i32
    // CHECK-NEXT: ttng.async_tma_store_wait {pendings = 0 : i32}
    // CHECK-NEXT: ttg.local_store
    ttg.local_store %zero, %src : tensor<16x16xf32, #blocked> -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @coalesce_adjacent_tle_stores
  tt.func public @coalesce_adjacent_tle_stores(%desc: !tt.tensordesc<tensor<16x16xf32, #shared>>, %src0: !ttg.memdesc<16x16xf32, #shared, #smem, mutable>, %src1: !ttg.memdesc<16x16xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    // CHECK: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src0 {tle.tma_store_explicit_commit} : !tt.tensordesc<tensor<16x16xf32, #shared>>, !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    tle.tma_store.commit_group
    ttng.async_tma_store_wait {pendings = 0 : i32}
    // CHECK: ttng.fence_async_shared
    ttng.fence_async_shared {bCluster = false}
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src1 {tle.tma_store_explicit_commit} : !tt.tensordesc<tensor<16x16xf32, #shared>>, !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    // CHECK-NEXT: tle.tma_store.commit_group
    // CHECK-NEXT: ttng.async_tma_store_wait {pendings = 0 : i32}
    // CHECK-NEXT: ttg.local_dealloc
    ttg.local_dealloc %src0 : !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    // CHECK-NEXT: ttg.local_dealloc
    ttg.local_dealloc %src1 : !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @keep_newer_group_pending
  tt.func public @keep_newer_group_pending(%desc: !tt.tensordesc<tensor<16x16xf32, #shared>>, %src0: !ttg.memdesc<16x16xf32, #shared, #smem, mutable>, %src1: !ttg.memdesc<16x16xf32, #shared, #smem, mutable>, %x: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    // CHECK: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src0 {tle.tma_store_explicit_commit} : !tt.tensordesc<tensor<16x16xf32, #shared>>, !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    tle.tma_store.commit_group
    ttng.async_tma_store_wait {pendings = 0 : i32}
    // CHECK-NEXT: tle.tma_store.commit_group
    // CHECK-NEXT: arith.addi
    %y = arith.addi %x, %c1 : i32
    // CHECK-NEXT: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src1 {tle.tma_store_explicit_commit} : !tt.tensordesc<tensor<16x16xf32, #shared>>, !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    tle.tma_store.commit_group
    ttng.async_tma_store_wait {pendings = 0 : i32}
    // CHECK-NEXT: tle.tma_store.commit_group
    // CHECK-NEXT: ttng.async_tma_store_wait {pendings = 1 : i32}
    // CHECK-NEXT: ttg.local_store
    ttg.local_store %zero, %src0 : tensor<16x16xf32, #blocked> -> !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    // CHECK-NEXT: ttng.async_tma_store_wait {pendings = 0 : i32}
    // CHECK-NEXT: tt.return
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @wait_before_lowered_pipe_release
  tt.func public @wait_before_lowered_pipe_release(%desc: !tt.tensordesc<tensor<16x16xf32, #shared>>, %src: !ttg.memdesc<16x16xf32, #shared, #smem, mutable>, %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, %x: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    // CHECK: ttng.async_tma_copy_local_to_global
    ttng.async_tma_copy_local_to_global %desc[%c0, %c0] %src {tle.tma_store_explicit_commit} : !tt.tensordesc<tensor<16x16xf32, #shared>>, !ttg.memdesc<16x16xf32, #shared, #smem, mutable>
    tle.tma_store.commit_group
    ttng.async_tma_store_wait {pendings = 0 : i32}
    // CHECK-NEXT: tle.tma_store.commit_group
    // CHECK-NEXT: arith.addi
    %y = arith.addi %x, %c1 : i32
    // CHECK-NEXT: ttng.async_tma_store_wait {pendings = 0 : i32}
    // CHECK-NEXT: ttng.arrive_barrier
    ttng.arrive_barrier %barrier, 128 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
    tt.return
  }
}
