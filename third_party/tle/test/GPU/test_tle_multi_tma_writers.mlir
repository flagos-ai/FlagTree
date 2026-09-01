// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: env TLE_MULTI_TMA_WRITERS=1 triton-opt --triton-tle-lower-pipe-to-nvws %s | FileCheck %s --check-prefix=NVWS
// RUN: env TLE_MULTI_TMA_WRITERS=1 triton-opt --triton-tle-lower-pipe-to-nvws --nvgpu-test-ws-lower-token %s | FileCheck %s --check-prefix=LOWER
// RUN: not env -u TLE_MULTI_TMA_WRITERS triton-opt --triton-tle-lower-pipe-to-nvws %s 2>&1 | FileCheck %s --check-prefix=DISABLED

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // DISABLED: mixed TMA/local-store pipe commit requires proven local-store writes
  // NVWS-LABEL: @multi_tma_writers
  // NVWS: %[[TOKEN:.*]] = nvws.create_token
  // NVWS-SAME: empty_count = 128 : i32
  // NVWS-SAME: full_count = 2 : i32
  // NVWS-SAME: loadType = 2 : i32
  // NVWS: nvws.producer_acquire %[[TOKEN]]{{.*}}async_task_id = array<i32: 0>
  // NVWS: nvws.producer_commit %[[TOKEN]]{{.*}}async_task_id = array<i32: 0>
  // NVWS: nvws.producer_acquire %[[TOKEN]]{{.*}}async_task_id = array<i32: 1>
  // NVWS: nvws.producer_commit %[[TOKEN]]{{.*}}async_task_id = array<i32: 1>
  // NVWS: nvws.consumer_wait %[[TOKEN]]{{.*}}async_task_id = array<i32: 2>

  // LOWER-LABEL: @multi_tma_writers
  // LOWER: ttng.init_barrier {{.*}}, 2
  // LOWER: ttng.init_barrier {{.*}}, 128
  // LOWER: ttng.init_barrier {{.*}}, 2
  // LOWER: ttng.init_barrier {{.*}}, 128
  // LOWER: ttng.barrier_expect {{.*}}, 8192
  // LOWER: ttng.async_tma_copy_global_to_local
  // LOWER: ttng.barrier_expect {{.*}}, 8192
  // LOWER: ttng.async_tma_copy_global_to_local
  tt.func @multi_tma_writers(
      %desc_a: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %desc_b: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>,
      %b: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a, %b {capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %a_slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_a, %a_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %b_slot = ttg.memdesc_index %b[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_b, %b_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    %closed = tle.pipe.reader_wait %a, %b[%c0, %false] {async_task_id = array<i32: 2>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    scf.if %closed {
    }
    tle.pipe.reader_release %a, %b[%c0] {async_task_id = array<i32: 2>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}
