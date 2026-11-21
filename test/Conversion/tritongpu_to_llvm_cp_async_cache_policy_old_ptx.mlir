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

// RUN: triton-opt %s -split-input-file -verify-diagnostics --allocate-shared-memory-nv --convert-triton-gpu-to-llvm="ptx-version=87" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @unsupported(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK: cp.async.cg.shared.global
    // CHECK-NOT: L2::cache_hint
    // expected-remark @+1 {{cache hint on NVIDIA cp.async:}}
    %0 = ttg.async_copy_global_to_local %src, %dst evictionPolicy = evict_first {contiguity = 4 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @unsupported(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK: cp.async.cg.shared.global
    // CHECK-NOT: L2::cache_hint
    // expected-remark @+1 {{cache hint on NVIDIA cp.async:}}
    %0 = ttg.async_copy_global_to_local %src, %dst evictionPolicy = evict_last {contiguity = 4 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @unsupported(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK: cp.async.cg.shared.global
    // CHECK-NOT: L2::cache_hint
    // expected-remark @+1 {{cache hint on NVIDIA cp.async:}}
    %0 = ttg.async_copy_global_to_local %src, %dst evictionPolicy = l2_evict_last {contiguity = 4 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}
