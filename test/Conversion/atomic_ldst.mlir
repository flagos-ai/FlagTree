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

// RUN: triton-opt %s --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 2>&1 | FileCheck %s --check-prefix=CHECK-TTG2NVGPU
// RUN: triton-opt %s --allocate-shared-memory-nv=compute-capability=90 --convert-triton-gpu-to-llvm=compute-capability=90 --convert-nv-gpu-to-llvm 2>&1 | FileCheck %s --check-prefix=CHECK-NVGPU2LLVM
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel_r(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c128_i32 : i32
    %2 = arith.cmpi slt, %1, %c512_i32 : i32

    // CHECK-TTG2NVGPU: nvg.ld_acquire acquire, gpu
    // CHECK-NVGPU2LLVM: ld.global.gpu.acquire.b32
    %3 = tt.atomic_rmw fadd, acquire, gpu, %arg0, %cst, %2 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %arg0, %3 : !tt.ptr<f32>

    // CHECK-TTG2NVGPU: nvg.ld_acquire acquire, cta
    // CHECK-NVGPU2LLVM: ld.global.cta.acquire.b32
    %4 = tt.atomic_rmw fadd, acquire, cta, %arg0, %cst, %true : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %arg0, %4 : !tt.ptr<f32>

    // CHECK-TTG2NVGPU: nvg.ld_acquire acquire, sys
    // CHECK-NVGPU2LLVM: ld.global.sys.acquire.b32
    %5 = tt.atomic_rmw fadd, acquire, sys, %arg0, %cst, %2 : (!tt.ptr<f32>, f32, i1) -> f32
    tt.store %arg0, %5 : !tt.ptr<f32>
    tt.return
  }
}
