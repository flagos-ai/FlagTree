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

// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %s --check-prefixes=CHECK,PTX81
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=94' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=94' -reconcile-unrealized-casts | FileCheck %s --check-prefixes=CHECK,PTX94

// The explicit zero fill stays asynchronous. NVIDIA's pipeline enhancement
// emits the eviction hint only with PTX 9.4 or newer and selects ca for this
// four-byte copy, even when the source requests cg (which requires 16 bytes).

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @async_copy_eviction_policy
  // PTX81-NOT: createpolicy
  // PTX81: cp.async.ca.shared.global [ $0 + 0 ], [ $1 + 0 ], 0x4,
  // PTX81-NOT: L2::cache_hint
  // PTX94: createpolicy.fractional.L2::evict_last.b64
  // PTX94: cp.async.ca.shared.global.L2::cache_hint [ $0 + 0 ], [ $1 + 0 ], 0x4,
  // CHECK-NOT: st.shared
  // CHECK-NOT: L2::256B
  tt.func @async_copy_eviction_policy(%gptr: tensor<1x4x!tt.ptr<f32>, #blocked>) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<1x4xf32, #blocked>
    %mask = arith.constant dense<true> : tensor<1x4xi1, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1x4xf32, #shared, #smem, mutable>
    ttg.async_copy_global_to_local %gptr, %smem mask %mask other %c0 cacheModifier = cg evictionPolicy = evict_last : tensor<1x4x!tt.ptr<f32>, #blocked> -> <1x4xf32, #shared, #smem, mutable>
    tt.return
  }
}
