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

// RUN: triton-opt %s -split-input-file -verify-diagnostics --allocate-shared-memory-nv --convert-triton-gpu-to-llvm="ptx-version=94" | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cg4
  tt.func @cg4(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: L1::evict
    // CHECK-NOT: L2::cache_hint
    // CHECK: cp.async.ca.shared.global
    // expected-remark @+1 {{cache hint on NVIDIA cp.async:}}
    %0 = ttg.async_copy_global_to_local %src, %dst cacheModifier = cg {contiguity = 1 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cg8
  tt.func @cg8(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: L1::evict
    // CHECK-NOT: L2::cache_hint
    // CHECK: cp.async.ca.shared.global
    // expected-remark @+1 {{cache hint on NVIDIA cp.async:}}
    %0 = ttg.async_copy_global_to_local %src, %dst cacheModifier = cg {contiguity = 2 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cs16
  tt.func @cs16(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: L1::evict
    // CHECK-NOT: L2::cache_hint
    // CHECK: cp.async.cg.shared.global
    // expected-remark @+1 {{cache hint on NVIDIA cp.async:}}
    %0 = ttg.async_copy_global_to_local %src, %dst cacheModifier = cs {contiguity = 4 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cs4
  tt.func @cs4(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: L1::evict
    // CHECK-NOT: L2::cache_hint
    // CHECK: cp.async.ca.shared.global
    // expected-remark @+1 {{cache hint on NVIDIA cp.async:}}
    %0 = ttg.async_copy_global_to_local %src, %dst cacheModifier = cs {contiguity = 1 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @lu16
  tt.func @lu16(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: L1::evict
    // CHECK-NOT: L2::cache_hint
    // CHECK: cp.async.cg.shared.global
    // expected-remark @+1 {{cache hint on NVIDIA cp.async:}}
    %0 = ttg.async_copy_global_to_local %src, %dst cacheModifier = lu {contiguity = 4 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @cv16
  tt.func @cv16(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: L1::evict
    // CHECK-NOT: L2::cache_hint
    // CHECK: cp.async.cg.shared.global
    // expected-remark @+1 {{cache hint on NVIDIA cp.async:}}
    %0 = ttg.async_copy_global_to_local %src, %dst cacheModifier = cv {contiguity = 4 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @l1_first
  tt.func @l1_first(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: L1::evict
    // CHECK-NOT: L2::cache_hint
    // CHECK: cp.async.cg.shared.global
    // expected-remark @+1 {{cache hint on NVIDIA cp.async:}}
    %0 = ttg.async_copy_global_to_local %src, %dst evictionPolicy = l1_evict_first {contiguity = 4 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @l1_last
  tt.func @l1_last(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: L1::evict
    // CHECK-NOT: L2::cache_hint
    // CHECK: cp.async.cg.shared.global
    // expected-remark @+1 {{cache hint on NVIDIA cp.async:}}
    %0 = ttg.async_copy_global_to_local %src, %dst evictionPolicy = l1_evict_last {contiguity = 4 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }
}
