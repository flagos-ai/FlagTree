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

// RUN: triton-opt --split-input-file -convert-proton-to-protongpu="max-shared-mem-size=32768" -proton-schedule-buffer-store -canonicalize -cse %s | FileCheck %s

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: simple_record
  // CHECK: %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
  // CHECK-NEXT: proton_gpu.initialize %[[SCRATCH]] : !tt.ptr<i32>
  // CHECK-NEXT: %[[BUF:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
  // CHECK-NEXT: %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[BUF]]
  // CHECK-NEXT: %[[START:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: %[[END:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: proton_gpu.circular_store start %[[SEGMENT]], %[[START]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: proton_gpu.circular_store end %[[SEGMENT]], %[[END]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: proton_gpu.finalize %[[SEGMENT]], %[[SCRATCH]] : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
  // CHECK-NEXT: tt.return
  tt.func @simple_record() {
    proton.record start "name0"
    proton.record end "name0"
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: simple_record
  // CHECK: %[[SCRATCH:.*]] = proton_gpu.global_scratch_alloc {alignment = 128 : i32, nbytes = 1152 : i32} : !tt.ptr<i32>
  // CHECK-NEXT: proton_gpu.initialize %[[SCRATCH]] : !tt.ptr<i32>
  // CHECK-NEXT: %[[BUF:.*]] = ttg.local_alloc  : () -> !ttg.memdesc<256xi32, #shared, #smem, mutable>
  // CHECK-NEXT: %[[SEGMENT:.*]] = proton_gpu.segment_alloc %[[BUF]]
  // CHECK-NEXT: %[[START1:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: %[[START2:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: %[[END2:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: proton_gpu.circular_store start %[[SEGMENT]], %[[START2]] {scopeId = 1 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: proton_gpu.circular_store end %[[SEGMENT]], %[[END2]] {scopeId = 1 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: %[[END1:.*]] = proton_gpu.read_counter : i32
  // CHECK-NEXT: proton_gpu.circular_store start %[[SEGMENT]], %[[START1]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: proton_gpu.circular_store end %[[SEGMENT]], %[[END1]] {scopeId = 0 : i32} : !proton_gpu.segment<1024, #smem, warp>, i32
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: proton_gpu.finalize %[[SEGMENT]], %[[SCRATCH]] : !proton_gpu.segment<1024, #smem, warp>, !tt.ptr<i32>
  // CHECK-NEXT: tt.return
  tt.func @simple_record() {
    proton.record start "name0"
    proton.record start "name1"
    proton.record end "name1"
    proton.record end "name0"
    tt.return
  }
}
