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

// RUN: triton-opt %s --allocate-shared-memory-nv --convert-triton-gpu-to-llvm="ptx-version=94" -reconcile-unrealized-casts | FileCheck %s

// Explicit modifiers and eviction policies survive final vectorization.
// Unsupported controls and old PTX versions are tested in the error fixtures.

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // The legacy eviction policy is a fractional L2 policy with fraction 1.0.
  // CHECK-LABEL: async_cp_evict_last_16_bytes
  tt.func @async_cp_evict_last_16_bytes(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK: createpolicy.fractional.L2::evict_last.b64 $0, 1.0;
    // CHECK: cp.async.cg.shared.global.L2::cache_hint [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x10, 0x10, ${{[0-9]+}};
    %0 = ttg.async_copy_global_to_local %src, %dst evictionPolicy = evict_last {contiguity = 4 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: async_cp_evict_first_masked_8_bytes
  tt.func @async_cp_evict_first_masked_8_bytes(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>, %valid: i1) {
    // CHECK: createpolicy.fractional.L2::evict_first.b64 $0, 1.0;
    // CHECK: cp.async.ca.shared.global.L2::cache_hint [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x8, ${{[0-9]+}}, ${{[0-9]+}};
    %mask = tt.splat %valid : i1 -> tensor<256xi1, #blocked>
    %0 = ttg.async_copy_global_to_local %src, %dst mask %mask evictionPolicy = evict_first {contiguity = 2 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }

  // `ca` is honored even for 16-byte copies, which default to `cg`.
  // CHECK-LABEL: async_cp_ca_16_bytes
  tt.func @async_cp_ca_16_bytes(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK-NOT: createpolicy
    // CHECK: cp.async.ca.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x10, 0x10;
    %0 = ttg.async_copy_global_to_local %src, %dst cacheModifier = ca {contiguity = 4 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: async_cp_cg_16_bytes
  tt.func @async_cp_cg_16_bytes(%src: tensor<256x!tt.ptr<f32>, #blocked>, %dst: !ttg.memdesc<256xf32, #shared, #smem, mutable>) {
    // CHECK: cp.async.cg.shared.global [ ${{.*}} + 0 ], [ ${{.*}} + 0 ], 0x10, 0x10;
    %0 = ttg.async_copy_global_to_local %src, %dst cacheModifier = cg {contiguity = 4 : i32} : tensor<256x!tt.ptr<f32>, #blocked> -> <256xf32, #shared, #smem, mutable>
    tt.return
  }

}
