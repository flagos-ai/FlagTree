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

// RUN: triton-opt %s -split-input-file -verify-diagnostics -triton-tle-downgrade-invalid-async-copy | FileCheck %s


#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
// CHECK-LABEL: @ordinary_unmasked
tt.func @ordinary_unmasked(%ptrs: tensor<64x128x!tt.ptr<f32>, #blocked>, %view: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
  // CHECK: ttg.async_copy_global_to_local {{.*}}evictionPolicy = evict_last
  %0 = ttg.async_copy_global_to_local %ptrs, %view evictionPolicy = evict_last : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
  // CHECK: ttg.async_copy_global_to_local {{.*}}evictionPolicy = l2_evict_last
  %1 = ttg.async_copy_global_to_local %ptrs, %view evictionPolicy = l2_evict_last : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
  tt.return
}
tt.func @omit_loop_free_partition_eviction_policy(%ptrs: tensor<64x128x!tt.ptr<f32>, #blocked>, %view: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
  ttg.warp_specialize(%view, %ptrs) attributes {requestedRegisters = array<i32: 72>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>, %arg1: tensor<64x128x!tt.ptr<f32>, #blocked>) num_warps(4) {
    // CHECK: ttg.async_copy_global_to_local
    // CHECK-NOT: evictionPolicy
    // expected-remark @+1 {{L2 hint omitted for unmasked or loop-free warp-specialize TLE local-pointer copy}}
    %token = ttg.async_copy_global_to_local %arg1, %arg0 evictionPolicy = evict_last {tle.local_ptr_async_store} : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<64x128xf32, #shared, #smem, mutable>, tensor<64x128x!tt.ptr<f32>, #blocked>) -> ()
  tt.return
}
}


// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "ppu:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @preserve_ppu_hint_until_backend_lowering(%ptrs: tensor<64x128x!tt.ptr<f32>, #blocked>, %view: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
  ttg.warp_specialize(%view, %ptrs) attributes {requestedRegisters = array<i32: 72>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>, %arg1: tensor<64x128x!tt.ptr<f32>, #blocked>) num_warps(4) {
    // CHECK: ttg.async_copy_global_to_local
    // CHECK-SAME: cacheModifier = ca
    %token = ttg.async_copy_global_to_local %arg1, %arg0 cacheModifier = ca {tle.local_ptr_async_store} : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<64x128xf32, #shared, #smem, mutable>, tensor<64x128x!tt.ptr<f32>, #blocked>) -> ()
  tt.return
}
}
