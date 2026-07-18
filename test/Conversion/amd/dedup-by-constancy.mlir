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

// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx942 --convert-builtin-func-to-llvm | FileCheck %s

// CHECK-LABEL: dedup_by_constancy_mfma
// CHECK-COUNT-2: llvm.icmp "slt"
// CHECK-NOT: llvm.icmp "slt"
// For a 32x32 tensor A with mfma layout, each thread holds 16 elements, which are divided
// into 4 groups. E.g. thread 0 holds elements A[0:3,0], A[8:11,0], A[16:19,0], and A[24:27,0].
// In this example, constancy of the tensor is 16 for dim 0, meaning A[0:15,0] have same values
// and A[16:31,0] have same values. Therefore, for thread 0, the first 8 elements are duplicated
// and the last 8 elements are duplicated.
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [1, 1], instrShape = [32, 32, 8], isTransposed = false}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @dedup_by_constancy_mfma(%arg0: i32 {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) {
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %1 = tt.splat %arg0 : i32 -> tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %2 = arith.cmpi slt, %0, %1 : tensor<32xi32, #ttg.slice<{dim = 1, parent = #mma}>>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<32xi1, #ttg.slice<{dim = 1, parent = #mma}>> -> tensor<32x1xi1, #mma>
    %4 = tt.broadcast %3 : tensor<32x1xi1, #mma> -> tensor<32x32xi1, #mma>
    %cst = arith.constant dense<0.100000e+00> : tensor<32x32xf16, #mma>
    %5 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x1x!tt.ptr<f16>, #mma>
    %6 = tt.broadcast %5 : tensor<32x1x!tt.ptr<f16>, #mma> -> tensor<32x32x!tt.ptr<f16>, #mma>
    tt.store %6, %cst, %4 : tensor<32x32x!tt.ptr<f16>, #mma>
    tt.return
  }
}
