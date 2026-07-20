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

// RUN: triton-opt %s -tritongpu-assign-latencies=num-stages=2 | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @pipeline_wgmma_view
  // CHECK: %[[LOAD:.+]] = tt.load {{.*}} {tt.latency = 1 : i32}
  // CHECK: ttg.local_alloc %[[LOAD]]
  tt.func @pipeline_wgmma_view(%base: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %a: !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %offs_m = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_k = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %m = tt.expand_dims %offs_m {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %k = tt.expand_dims %offs_k {axis = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x512xi32, #blocked>
    %stride = arith.constant dense<512> : tensor<64x1xi32, #blocked>
    %m_off = arith.muli %m, %stride : tensor<64x1xi32, #blocked>
    %m_off_b = tt.broadcast %m_off : tensor<64x1xi32, #blocked> -> tensor<64x512xi32, #blocked>
    %k_b = tt.broadcast %k : tensor<1x512xi32, #blocked> -> tensor<64x512xi32, #blocked>
    %offs = arith.addi %m_off_b, %k_b : tensor<64x512xi32, #blocked>
    %ptr = tt.splat %base : !tt.ptr<bf16> -> tensor<64x512x!tt.ptr<bf16>, #blocked>
    %ptrs = tt.addptr %ptr, %offs : tensor<64x512x!tt.ptr<bf16>, #blocked>, tensor<64x512xi32, #blocked>
    %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %cst) -> tensor<64x64xf32, #mma> {
      %b = tt.load %ptrs : tensor<64x512x!tt.ptr<bf16>, #blocked>
      %b_smem = ttg.local_alloc %b : (tensor<64x512xbf16, #blocked>) -> !ttg.memdesc<64x512xbf16, #shared, #smem, mutable>
      %b_view = tle.memdesc_wgmma_view %b_smem {order = array<i32: 1, 0>} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable>
      %next = ttng.warp_group_dot %a, %b_view, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x512xbf16, #shared, #smem, mutable> * !ttg.memdesc<512x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      scf.yield %next : tensor<64x64xf32, #mma>
    } {tt.num_stages = 2 : i32}
    tt.return %out : tensor<64x64xf32, #mma>
  }
}
