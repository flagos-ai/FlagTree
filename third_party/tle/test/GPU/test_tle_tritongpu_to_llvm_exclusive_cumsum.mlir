// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=120 ptx-version=88}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=120 ptx-version=88}, canonicalize, cse, convert-nv-gpu-to-llvm, convert-warp-specialize-to-llvm, canonicalize, cse, symbol-dce, convert-nvvm-to-llvm)' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cumsum_to_llvm(%arg0: tensor<128xi32, #blocked>, %out: !tt.ptr<i32>) {
    %exclusive, %total = "tle.exclusive_cumsum"(%arg0) {axis = 0 : i32, reverse = false} : (tensor<128xi32, #blocked>) -> (tensor<128xi32, #blocked>, i32)
    tt.store %out, %total : !tt.ptr<i32>
    tt.return
  }
}

// CHECK: llvm.func @cumsum_to_llvm
// CHECK-NOT: tle.exclusive_cumsum
