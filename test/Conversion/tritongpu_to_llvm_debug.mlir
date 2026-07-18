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

// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-gpu-to-llvm --debug| FileCheck %s

// CHECK-LABEL: convert_identity
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @convert_identity(%arg0: tensor<128x128xf16, #blocked>) {
    %1 = ttg.convert_layout %arg0 : tensor<128x128xf16, #blocked> -> tensor<128x128xf16, #blocked>
    tt.return
  }
}
