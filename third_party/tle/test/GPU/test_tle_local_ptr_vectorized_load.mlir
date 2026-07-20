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

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [16, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @local_ptr_v4_load() {
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant dense<4> : tensor<512x4xi32, #blocked>
    %smem = ttg.local_alloc {tle.barrier_group = 0 : i64} : () -> !ttg.memdesc<4096xi32, #shared, #smem, mutable>
    %base = "tle.local_pointers"(%smem, %c0_i32) {tle.barrier_group = 0 : i64} : (!ttg.memdesc<4096xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %basev = tt.splat %base : !tt.ptr<i32, 3> -> tensor<512x4x!tt.ptr<i32, 3>, #blocked>

    %row = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<512x1xi32, #blocked>
    %rowb = tt.broadcast %row2d : tensor<512x1xi32, #blocked> -> tensor<512x4xi32, #blocked>
    %col = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col2d = tt.expand_dims %col {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi32, #blocked>
    %colb = tt.broadcast %col2d : tensor<1x4xi32, #blocked> -> tensor<512x4xi32, #blocked>
    %row_scaled = arith.muli %rowb, %c4 : tensor<512x4xi32, #blocked>
    %offs = arith.addi %row_scaled, %colb : tensor<512x4xi32, #blocked>

    %ptrs = tt.addptr %basev, %offs : tensor<512x4x!tt.ptr<i32, 3>, #blocked>, tensor<512x4xi32, #blocked>
    %vals = tt.load %ptrs : tensor<512x4x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }
}

// CHECK: ld.shared.v4.b32
