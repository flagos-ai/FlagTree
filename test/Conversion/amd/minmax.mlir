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

// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx942 | FileCheck %s --check-prefix=GFX942
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck %s --check-prefix=GFX950

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {

// GFX942: llvm.func @min_max
// GFX942-COUNT-2: llvm.fcmp
// GFX942: llvm.or
// GFX942: llvm.intr.minnum
// GFX942-COUNT-2: llvm.fcmp
// GFX942: llvm.or
// GFX942: llvm.intr.maxnum

// GFX950: llvm.func @min_max
// GFX950: llvm.intr.minimum
// GFX950-NEXT: llvm.intr.maximum
  tt.func public @min_max(%arg0: f32, %arg1: f32) {
    %0 = arith.minimumf %arg0, %arg1 : f32
    %1 = arith.maximumf %arg0, %arg1 : f32
    tt.return
  }
}
