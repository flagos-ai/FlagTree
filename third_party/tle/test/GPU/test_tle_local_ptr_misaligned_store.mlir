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

// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=120 ptx-version=88}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=120 ptx-version=88}, canonicalize, cse, convert-nv-gpu-to-llvm, convert-warp-specialize-to-llvm, canonicalize, cse, symbol-dce, convert-nvvm-to-llvm)' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [16], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // Layout-derived vector hints are legal only when AxisInfo divisibility proves
  // that the first element is aligned for the requested vector width. This
  // tensor-indexed local_ptr starts at offset 1, so it is only 4-byte aligned
  // and must not lower to st.shared.v4.b32.
  tt.func public @misaligned_tensor_indexed_local_ptr_store() {
    %one = arith.constant dense<1> : tensor<128xi32, #blocked>
    %offs0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %offs = arith.addi %offs0, %one : tensor<128xi32, #blocked>
    %vals = arith.constant dense<0.000000e+00> : tensor<128xf32, #blocked>
    %smem = ttg.local_alloc {tle.barrier_group = 0 : i64} : () -> !ttg.memdesc<256xf32, #shared, #smem, mutable>
    %ptrs = "tle.local_pointers"(%smem, %offs) {tle.barrier_group = 0 : i64} : (!ttg.memdesc<256xf32, #shared, #smem, mutable>, tensor<128xi32, #blocked>) -> tensor<128x!tt.ptr<f32, 3>, #blocked>
    tt.store %ptrs, %vals : tensor<128x!tt.ptr<f32, 3>, #blocked>
    tt.return
  }
}

// CHECK-LABEL: @misaligned_tensor_indexed_local_ptr_store
// CHECK-NOT: st.shared.v4.b32
// CHECK: st.shared.b32
