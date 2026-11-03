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

// RUN: triton-opt %s -split-input-file -triton-tle-downgrade-invalid-async-copy | FileCheck %s

// The copy becomes a load and a store. Its commit group and wait stay, now
// without the token, so the commit groups keep their numbering.
// CHECK-LABEL: tt.func @downgrade_bf16
// CHECK-NOT: ttg.async_copy_global_to_local
// CHECK: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<32x512x!tt.ptr<bf16>, #{{.*}}>
// CHECK: ttg.local_store %[[LOAD]], %{{.*}} : tensor<32x512xbf16, #{{.*}}> -> !ttg.memdesc<32x512xbf16
// CHECK: %[[COMMIT:.*]] = ttg.async_commit_group{{$}}
// CHECK: %[[WAIT:.*]] = ttg.async_wait %[[COMMIT]] {num = 0 : i32}
// CHECK: %{{.*}} = ttg.local_load %{{.*}} token %[[WAIT]] : !ttg.memdesc<32x512xbf16
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
tt.func @downgrade_bf16(%input: tensor<32x512x!tt.ptr<bf16>, #blocked>,
                        %view: !ttg.memdesc<32x512xbf16, #shared, #smem, mutable>)
                        -> tensor<32x512xbf16, #blocked> {
  %token = ttg.async_copy_global_to_local %input, %view : tensor<32x512x!tt.ptr<bf16>, #blocked> -> <32x512xbf16, #shared, #smem, mutable>
  %commit = ttg.async_commit_group tokens %token
  %wait = ttg.async_wait %commit {num = 0 : i32}
  %loaded = ttg.local_load %view token %wait : !ttg.memdesc<32x512xbf16, #shared, #smem, mutable> -> tensor<32x512xbf16, #blocked>
  tt.return %loaded : tensor<32x512xbf16, #blocked>
}
}

// -----

// CHECK-LABEL: tt.func @coalesced_bf16_fallback
// CHECK: %[[PTR_CVT:.*]] = ttg.convert_layout %{{.*}} : tensor<64x512x!tt.ptr<bf16>, #{{.*}}> -> tensor<64x512x!tt.ptr<bf16>, #[[VEC:.*]]>
// CHECK: %[[LOAD:.*]] = tt.load %[[PTR_CVT]] : tensor<64x512x!tt.ptr<bf16>, #[[VEC]]>
// CHECK: %[[VAL_CVT:.*]] = ttg.convert_layout %[[LOAD]] : tensor<64x512xbf16, #[[VEC]]> -> tensor<64x512xbf16, #{{.*}}>
// CHECK: ttg.local_store %[[VAL_CVT]], %{{.*}} : tensor<64x512xbf16, #{{.*}}> -> !ttg.memdesc<64x512xbf16
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
tt.func @coalesced_bf16_fallback(
    %ptrs: tensor<64x512x!tt.ptr<bf16>, #blocked> {tt.contiguity = dense<[1, 512]> : tensor<2xi32>, tt.divisibility = dense<[1, 16]> : tensor<2xi32>},
    %view: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) {
  %token = ttg.async_copy_global_to_local %ptrs, %view {tle.local_ptr_async_store} : tensor<64x512x!tt.ptr<bf16>, #blocked> -> <64x512xbf16, #shared, #smem, mutable>
  %commit = ttg.async_commit_group tokens %token
  ttg.async_wait %commit {num = 0 : i32}
  tt.return
}
}

// -----

// CHECK-LABEL: tt.func @drop_loop_free_partition_eviction_policy
// CHECK: partition0
// CHECK: ttg.async_copy_global_to_local
// CHECK-SAME: {tle.local_ptr_async_store}
// CHECK-NOT: evictionPolicy = evict_last
// CHECK: ttg.warp_return
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @drop_loop_free_partition_eviction_policy(%ptrs: tensor<64x128x!tt.ptr<f32>, #blocked>, %view: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
  ttg.warp_specialize(%view, %ptrs) attributes {requestedRegisters = array<i32: 72>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>, %arg1: tensor<64x128x!tt.ptr<f32>, #blocked>) num_warps(4) {
    %token = ttg.async_copy_global_to_local %arg1, %arg0 evictionPolicy = evict_last {tle.local_ptr_async_store} : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
    ttg.warp_return
  } : (!ttg.memdesc<64x128xf32, #shared, #smem, mutable>, tensor<64x128x!tt.ptr<f32>, #blocked>) -> ()
  tt.return
}
}

// -----

// CHECK-LABEL: tt.func @preserve_loop_partition_eviction_policy
// CHECK: partition0
// CHECK: scf.for
// CHECK: ttg.async_copy_global_to_local
// CHECK-SAME: evictionPolicy = evict_last
// CHECK-SAME: {tle.local_ptr_async_store}
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @preserve_loop_partition_eviction_policy(%ptrs: tensor<64x128x!tt.ptr<f32>, #blocked>, %view: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>) {
  ttg.warp_specialize(%view, %ptrs) attributes {requestedRegisters = array<i32: 72>}
  default {
    ttg.warp_yield
  }
  partition0(%arg0: !ttg.memdesc<64x128xf32, #shared, #smem, mutable>, %arg1: tensor<64x128x!tt.ptr<f32>, #blocked>) num_warps(4) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %token = ttg.async_copy_global_to_local %arg1, %arg0 evictionPolicy = evict_last {tle.local_ptr_async_store} : tensor<64x128x!tt.ptr<f32>, #blocked> -> <64x128xf32, #shared, #smem, mutable>
    }
    ttg.warp_return
  } : (!ttg.memdesc<64x128xf32, #shared, #smem, mutable>, tensor<64x128x!tt.ptr<f32>, #blocked>) -> ()
  tt.return
}
}

// -----

// A copy too narrow for cp.async can share its commit group with a copy that
// stays asynchronous. It leaves the group and the wait, which keep serving the
// copy that remains, instead of failing the pass.
// CHECK-LABEL: tt.func @downgrade_shared_commit_group
// CHECK-NOT:   ttg.async_commit_group tokens %{{.*}}, %{{.*}}
// CHECK:       %[[LOADED:.*]] = tt.load %{{.*}} : tensor<32x512x!tt.ptr<bf16>
// CHECK-NEXT:  ttg.local_store %[[LOADED]]
// CHECK:       %[[VALID:.*]] = ttg.async_copy_global_to_local
// CHECK:       %[[COMMIT:.*]] = ttg.async_commit_group tokens %[[VALID]]
// CHECK:       %[[WAIT:.*]] = ttg.async_wait %[[COMMIT]]
// CHECK:       ttg.local_load %{{.*}} token %[[WAIT]]
// CHECK:       ttg.local_load %{{.*}} token %[[WAIT]]
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 8], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
tt.func @downgrade_shared_commit_group(%narrow: tensor<32x512x!tt.ptr<bf16>, #blocked>,
                                       %wide: tensor<32x512x!tt.ptr<f32>, #blocked>,
                                       %narrowView: !ttg.memdesc<32x512xbf16, #shared, #smem, mutable>,
                                       %wideView: !ttg.memdesc<32x512xf32, #shared, #smem, mutable>)
                                       -> (tensor<32x512xbf16, #blocked>, tensor<32x512xf32, #blocked>) {
  %narrowToken = ttg.async_copy_global_to_local %narrow, %narrowView : tensor<32x512x!tt.ptr<bf16>, #blocked> -> <32x512xbf16, #shared, #smem, mutable>
  %wideToken = ttg.async_copy_global_to_local %wide, %wideView : tensor<32x512x!tt.ptr<f32>, #blocked> -> <32x512xf32, #shared, #smem, mutable>
  %commit = ttg.async_commit_group tokens %narrowToken, %wideToken
  %wait = ttg.async_wait %commit {num = 0 : i32}
  %narrowLoaded = ttg.local_load %narrowView token %wait : !ttg.memdesc<32x512xbf16, #shared, #smem, mutable> -> tensor<32x512xbf16, #blocked>
  %wideLoaded = ttg.local_load %wideView token %wait : !ttg.memdesc<32x512xf32, #shared, #smem, mutable> -> tensor<32x512xf32, #blocked>
  tt.return %narrowLoaded, %wideLoaded : tensor<32x512xbf16, #blocked>, tensor<32x512xf32, #blocked>
}
}

// -----

// Every copy of a commit group is too narrow. The group stays, empty, between
// the groups of the other copies: the wait distance `num = 2` of the first
// wait still counts it, so the copy it waits for is complete when its data is
// read.
// CHECK-LABEL: tt.func @downgrade_whole_commit_group
// CHECK:       %[[FIRST:.*]] = ttg.async_commit_group tokens %{{.*}}
// CHECK:       tt.load %{{.*}} : tensor<128x!tt.ptr<f16>
// CHECK-NEXT:  ttg.local_store
// CHECK:       tt.load %{{.*}} : tensor<128x!tt.ptr<f16>
// CHECK-NEXT:  ttg.local_store
// CHECK:       %[[EMPTY:.*]] = ttg.async_commit_group{{$}}
// CHECK:       ttg.async_commit_group tokens %{{.*}}
// CHECK:       %[[WAIT_FIRST:.*]] = ttg.async_wait %[[FIRST]] {num = 2 : i32}
// CHECK:       ttg.local_load %{{.*}} token %[[WAIT_FIRST]]
// CHECK:       %[[WAIT_EMPTY:.*]] = ttg.async_wait %[[EMPTY]] {num = 1 : i32}
// CHECK:       ttg.local_load %{{.*}} token %[[WAIT_EMPTY]]
// CHECK:       ttg.local_load %{{.*}} token %[[WAIT_EMPTY]]
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @downgrade_whole_commit_group(%first: tensor<128x!tt.ptr<f32>, #blocked> {tt.contiguity = 4 : i32, tt.divisibility = 16 : i32},
                                      %narrowA: tensor<128x!tt.ptr<f16>, #blocked> {tt.contiguity = 1 : i32, tt.divisibility = 16 : i32},
                                      %narrowB: tensor<128x!tt.ptr<f16>, #blocked> {tt.contiguity = 1 : i32, tt.divisibility = 16 : i32},
                                      %last: tensor<128x!tt.ptr<f32>, #blocked> {tt.contiguity = 4 : i32, tt.divisibility = 16 : i32}) {
  %firstView = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
  %narrowViewA = ttg.local_alloc : () -> !ttg.memdesc<128xf16, #shared, #smem, mutable>
  %narrowViewB = ttg.local_alloc : () -> !ttg.memdesc<128xf16, #shared, #smem, mutable>
  %lastView = ttg.local_alloc : () -> !ttg.memdesc<128xf32, #shared, #smem, mutable>
  %firstToken = ttg.async_copy_global_to_local %first, %firstView : tensor<128x!tt.ptr<f32>, #blocked> -> <128xf32, #shared, #smem, mutable>
  %firstCommit = ttg.async_commit_group tokens %firstToken
  %narrowTokenA = ttg.async_copy_global_to_local %narrowA, %narrowViewA : tensor<128x!tt.ptr<f16>, #blocked> -> <128xf16, #shared, #smem, mutable>
  %narrowTokenB = ttg.async_copy_global_to_local %narrowB, %narrowViewB : tensor<128x!tt.ptr<f16>, #blocked> -> <128xf16, #shared, #smem, mutable>
  %narrowCommit = ttg.async_commit_group tokens %narrowTokenA, %narrowTokenB
  %lastToken = ttg.async_copy_global_to_local %last, %lastView : tensor<128x!tt.ptr<f32>, #blocked> -> <128xf32, #shared, #smem, mutable>
  %lastCommit = ttg.async_commit_group tokens %lastToken
  %firstWait = ttg.async_wait %firstCommit {num = 2 : i32}
  %firstLoaded = ttg.local_load %firstView token %firstWait : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
  %narrowWait = ttg.async_wait %narrowCommit {num = 1 : i32}
  %narrowLoadedA = ttg.local_load %narrowViewA token %narrowWait : !ttg.memdesc<128xf16, #shared, #smem, mutable> -> tensor<128xf16, #blocked>
  %narrowLoadedB = ttg.local_load %narrowViewB token %narrowWait : !ttg.memdesc<128xf16, #shared, #smem, mutable> -> tensor<128xf16, #blocked>
  %lastWait = ttg.async_wait %lastCommit {num = 0 : i32}
  %lastLoaded = ttg.local_load %lastView token %lastWait : !ttg.memdesc<128xf32, #shared, #smem, mutable> -> tensor<128xf32, #blocked>
  tt.return
}
}
