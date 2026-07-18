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

// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="arch=gfx942 ftz=True" --convert-builtin-func-to-llvm="ftz=True" | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="arch=gfx950 ftz=True" --convert-builtin-func-to-llvm="ftz=True" | FileCheck %s --check-prefix=LLVM_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="arch=gfx942 ftz=False" --convert-builtin-func-to-llvm="ftz=False" | FileCheck %s --check-prefix=LLVM_NO_FTZ
// RUN: triton-opt %s --convert-triton-amdgpu-to-llvm="arch=gfx950 ftz=False" --convert-builtin-func-to-llvm="ftz=False" | FileCheck %s --check-prefix=LLVM_NO_FTZ

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fast_expf(%arg0: tensor<64xf32, #blocked>) {
    // CHECK-LABEL: test_fast_expf
    // LLVM_FTZ: llvm.amdgcn.exp2.f32
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_expf"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return
  }
}

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @test_fast_tanhf(%arg0: tensor<64xf32, #blocked>) {
    // CHECK-LABEL: test_fast_tanhf
    // LLVM_FTZ: llvm.amdgcn.exp2.f32
    // LLVM_NO_FTZ: llvm.exp2.f32
    %0 = tt.extern_elementwise %arg0 {libname = "libdevice", libpath = "", pure = true, symbol = "__triton_hip_fast_tanhf"} : (tensor<64xf32, #blocked>) -> tensor<64xf32, #blocked>
    tt.return
  }
}
