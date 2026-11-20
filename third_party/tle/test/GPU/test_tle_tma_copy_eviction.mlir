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

// RUN: triton-opt %s -split-input-file -triton-tle-lower-tma-copy -verify-diagnostics | FileCheck %s

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar_slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @load_default
  tt.func @load_default(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>) {
    %c0 = arith.constant 0 : i32
    %dst = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // CHECK: ttng.async_tma_copy_global_to_local{{.*}} %true :
    ttg.tma_copy %desc, %dst, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar_slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @load_evict_first
  tt.func @load_evict_first(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>) {
    %c0 = arith.constant 0 : i32
    %dst = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}}evictionPolicy = evict_first
    ttg.tma_copy %desc, %dst, [%c0, %c0] evictionPolicy = evict_first : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar_slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @load_evict_last
  tt.func @load_evict_last(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>) {
    %c0 = arith.constant 0 : i32
    %dst = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}}evictionPolicy = evict_last
    ttg.tma_copy %desc, %dst, [%c0, %c0] evictionPolicy = evict_last : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar_slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @load_evict_first_explicit_barrier
  tt.func @load_evict_first_explicit_barrier(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %bar: !ttg.memdesc<1xi64, #bar_slot, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %dst = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}}evictionPolicy = evict_first
    ttg.tma_copy %desc, %dst, [%c0, %c0], barrier %bar evictionPolicy = evict_first {expect_bytes = 8192 : i32} : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<1xi64, #bar_slot, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar_slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @load_evict_last_explicit_barrier
  tt.func @load_evict_last_explicit_barrier(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %bar: !ttg.memdesc<1xi64, #bar_slot, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %dst = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}}evictionPolicy = evict_last
    ttg.tma_copy %desc, %dst, [%c0, %c0], barrier %bar evictionPolicy = evict_last {expect_bytes = 8192 : i32} : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<1xi64, #bar_slot, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar_slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func @reject_store_policy(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>) {
    %c0 = arith.constant 0 : i32
    %src = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{eviction policies are supported only for global-to-shared TMA copies}}
    ttg.tma_copy %src, %desc, [%c0, %c0] evictionPolicy = evict_last : !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>, !tt.tensordesc<tensor<32x64xf32, #nvmma>>
    tt.return
  }
}
