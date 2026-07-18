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

// RUN: triton-opt --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: proton_record
  tt.func @proton_record() {
    // CHECK: proton.record start "name0"
    // CHECK: proton.record end "name0"
    // CHECK-NEXT: tt.return
    proton.record start "name0"
    proton.record end "name0"
    tt.return
  }
} // end module

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: protongpu_ops
  tt.func @protongpu_ops() {
    // CHECK: ttg.local_alloc
    // CHECK-NEXT: proton_gpu.global_scratch_alloc
    // CHECK-NEXT: proton_gpu.initialize
    // CHECK-NEXT: proton_gpu.segment_alloc
    // CHECK-NEXT: proton_gpu.init_ctx
    // CHECK-NEXT: proton_gpu.read_counter
    // CHECK-NEXT: proton_gpu.circular_store start
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: proton_gpu.save_ctx
    // CHECK-NEXT: proton_gpu.finalize
    // CHECK-NEXT: tt.return
    %0 = ttg.local_alloc : () -> !ttg.memdesc<64xi32, #shared, #smem, mutable>
    %1 = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 384 : i32} : !tt.ptr<i32>
    proton_gpu.initialize %1 : !tt.ptr<i32>
    %seg = proton_gpu.segment_alloc %0 : !ttg.memdesc<64xi32, #shared, #smem, mutable> -> !proton_gpu.segment<256, #shared, warp>
    proton_gpu.init_ctx %1 : !tt.ptr<i32>
    %3 = proton_gpu.read_counter : i32
    proton_gpu.circular_store start %seg, %3 {scopeId = 0 : i32} : !proton_gpu.segment<256, #shared, warp>, i32
    gpu.barrier
    proton_gpu.save_ctx %seg, %1: !proton_gpu.segment<256, #shared, warp>, !tt.ptr<i32>
    proton_gpu.finalize %seg, %1 : !proton_gpu.segment<256, #shared, warp>, !tt.ptr<i32>
    tt.return
  }
} // end module
