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

// RUN: triton-opt %s -split-input-file --tritongpu-global-scratch-memory-allocation | FileCheck %s

// CHECK: module attributes {ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 256 : i32{{.*}}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK: @test_alloc{{.*}}ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 256 : i32
  tt.func public @test_alloc() -> (!tt.ptr<i8>, !tt.ptr<i8>) {
    // CHECK:  ttg.global_scratch_memory_offset = 0
    %0 = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32} : !tt.ptr<i8>
    // CHECK:  ttg.global_scratch_memory_offset = 128
    %1 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
    tt.return %0, %1 : !tt.ptr<i8>, !tt.ptr<i8>
  }
}

// -----

// CHECK: module attributes {ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 256 : i32{{.*}}}
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
// CHECK: @helper1{{.*}}ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 128 : i32
  tt.func private @helper1() -> (!tt.ptr<i8>) {
    // CHECK:  ttg.global_scratch_memory_offset = 0
    %0 = ttg.global_scratch_alloc {alignment = 128 : i32, nbytes = 128 : i32} : !tt.ptr<i8>
    tt.return %0 : !tt.ptr<i8>
  }

// CHECK: @test_function{{.*}}ttg.global_scratch_memory_alignment = 128 : i32, ttg.global_scratch_memory_size = 256 : i32
  tt.func public @test_function() -> (!tt.ptr<i8>, !tt.ptr<i8>) {
    // CHECK:  ttg.global_scratch_memory_offset = 0
    %0 = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32} : !tt.ptr<i8>
    // CHECK:  ttg.global_scratch_memory_offset = 128
    %1 = tt.call @helper1() : () -> (!tt.ptr<i8>)
    tt.return %0, %1 : !tt.ptr<i8>, !tt.ptr<i8>
  }
}
