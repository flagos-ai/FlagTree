/*
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

// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=90 ptx-version=80}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=80})' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @callee(%counter_remote_ptr: !tt.ptr<i32, 7>, %out_remote_ptr: !tt.ptr<i32, 7>) {
    %c0_i32 = arith.constant 0 : i32
    %lane = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %zeros = arith.constant dense<0> : tensor<128xi32, #blocked>
    %ones = arith.constant dense<1> : tensor<128xi32, #blocked>
    %pred = arith.constant dense<true> : tensor<128xi1, #blocked>
    %counter_ptrs = tt.splat %counter_remote_ptr : !tt.ptr<i32, 7> -> tensor<128x!tt.ptr<i32, 7>, #blocked>
    %counter_ptrs_2 = tt.addptr %counter_ptrs, %zeros : tensor<128x!tt.ptr<i32, 7>, #blocked>, tensor<128xi32, #blocked>
    %pos = tt.atomic_rmw add, relaxed, cta, %counter_ptrs_2, %ones, %pred : (tensor<128x!tt.ptr<i32, 7>, #blocked>, tensor<128xi32, #blocked>, tensor<128xi1, #blocked>) -> tensor<128xi32, #blocked>
    %out_ptrs = tt.splat %out_remote_ptr : !tt.ptr<i32, 7> -> tensor<128x!tt.ptr<i32, 7>, #blocked>
    %dst = tt.addptr %out_ptrs, %pos : tensor<128x!tt.ptr<i32, 7>, #blocked>, tensor<128xi32, #blocked>
    tt.store %dst, %lane : tensor<128x!tt.ptr<i32, 7>, #blocked>
    tt.return
  }

  tt.func public @caller() {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2048xi32, #shared, #smem, mutable>
    %counter_local_ptr = "tle.local_pointers"(%smem, %c0_i32) : (!ttg.memdesc<2048xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %out_local_ptr = "tle.local_pointers"(%smem, %c1_i32) : (!ttg.memdesc<2048xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %counter_remote_ptr = "tle.remote_pointers"(%counter_local_ptr, %c0_i32) : (!tt.ptr<i32, 3>, i32) -> !tt.ptr<i32, 7>
    %out_remote_ptr = "tle.remote_pointers"(%out_local_ptr, %c0_i32) : (!tt.ptr<i32, 3>, i32) -> !tt.ptr<i32, 7>
    tt.call @callee(%counter_remote_ptr, %out_remote_ptr) : (!tt.ptr<i32, 7>, !tt.ptr<i32, 7>) -> ()
    tt.return
  }
}

// CHECK: llvm.func internal @callee(%arg0: !llvm.ptr<7>
// CHECK-SAME: %arg1: !llvm.ptr<7>
// CHECK: atom.shared::cluster.cta.relaxed.add.u32
// CHECK: st.shared::cluster.b32
// CHECK: nvvm.mapa
