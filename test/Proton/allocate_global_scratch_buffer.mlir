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

// RUN: triton-opt --split-input-file -allocate-proton-global-scratch-buffer %s | FileCheck %s

// CHECK: module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.profile_scratch_memory_alignment = 128 : i32, ttg.profile_scratch_memory_size = 768 : i32} {
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func @test_empty_kernel(%lb : index, %A : !tt.ptr<i8>) {
    // CHECK: %0 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32, offset = 0 : i32} : !tt.ptr<i8>
    %0 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32} : !tt.ptr<i8>
    // CHECK: %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32, offset = 384 : i32} : !tt.ptr<i8>
    %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32} : !tt.ptr<i8>
    tt.return
  }
}
