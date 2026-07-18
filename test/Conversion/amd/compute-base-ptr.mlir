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

// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx942 --mlir-print-debuginfo --mlir-pretty-debuginfo| FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [4, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 4], instrShape = [16, 16, 16], isTransposed = false}>
#shared = #ttg.swizzled_shared<{vec = 16, perPhase = 4, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.shared = 544 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @local_load_offset
  tt.func @local_load_offset(%arg0: tensor<16x16xf16, #mma>) {
    %0 = ttg.convert_layout %arg0 {allocation.offset = 0 : i32} : tensor<16x16xf16, #mma> -> tensor<16x16xf16, #blocked> loc(#loc1)
    %1 = ttg.local_alloc %0 {allocation.offset = 0 : i32} : (tensor<16x16xf16, #blocked>) -> !ttg.memdesc<16x16xf16, #shared, #smem> loc(#loc2)
    // This catches base ptr calculation in the computeBasePtr, checks if the gep has correct element type.
    // CHECK: llvm.getelementptr {{.*}} (!llvm.ptr<3>, i32) -> !llvm.ptr<3>, i8 local_load:3:0
    %2 = ttg.local_load %1 : !ttg.memdesc<16x16xf16, #shared, #smem> -> tensor<16x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 16}>> loc(#loc3)
    tt.return
  }
}
#loc1 = loc("conert_layout":1:0)
#loc2 = loc("local_alloc":2:0)
#loc3 = loc("local_load":3:0)
