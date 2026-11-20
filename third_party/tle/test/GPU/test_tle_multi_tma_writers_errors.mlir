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

// RUN: triton-opt %s -triton-tle-lower-pipe-to-nvws -split-input-file -verify-diagnostics

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_partial_tma_with_one_writer(
      %desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>,
      %b: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a, %b {capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %a_slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %a_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{mixed TMA/local-store pipe commit requires proven local-store writes for the non-TMA fields}}
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_multi_writer_incomplete_shared_allocation_subslices(
      %desc_a: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %desc_b: !tt.tensordesc<tensor<32x64xf32, #nvmma>>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %storage = ttg.local_alloc : () -> !ttg.memdesc<2x128x64xf32, #shared3, #smem, mutable>
    %a = ttg.memdesc_subslice %storage[0, 0, 0] : !ttg.memdesc<2x128x64xf32, #shared3, #smem, mutable> -> !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>
    %b = ttg.memdesc_subslice %storage[0, 32, 0] : !ttg.memdesc<2x128x64xf32, #shared3, #smem, mutable> -> !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>
    %c = ttg.memdesc_subslice %storage[0, 64, 0] : !ttg.memdesc<2x128x64xf32, #shared3, #smem, mutable> -> !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>
    tle.pipe.create %a, %b, %c {capacity = 2 : i32, pipe_name = "incomplete_subviews", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>

    tle.pipe.writer_acquire %a, %b, %c[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "incomplete_subviews", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>
    %a_slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable, 2x128x64>
    ttg.tma_copy %desc_a, %a_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable, 2x128x64>
    tle.pipe.writer_commit %a, %b, %c[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "incomplete_subviews", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>

    tle.pipe.writer_acquire %a, %b, %c[%c0, %false] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "incomplete_subviews", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>
    %b_slot = ttg.memdesc_index %b[%c0] : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable, 2x128x64>
    ttg.tma_copy %desc_b, %b_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable, 2x128x64>
    // expected-error @+1 {{uses multiple pure-TMA writers whose combined commits do not cover every pipe field}}
    tle.pipe.writer_commit %a, %b, %c[%c0] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "incomplete_subviews", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x128x64>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_multi_writer_unbalanced_lifecycle_sites(
      %desc_a: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %desc_b: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>,
      %b: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a, %b {capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %a_slot0 = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_a, %a_slot0, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    // A second static contribution from task 0 has no peer contribution from
    // task 1, so full_count=2 would be wrong for this lifecycle site.
    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %a_slot1 = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_a, %a_slot1, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %b_slot = ttg.memdesc_index %b[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_b, %b_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{uses multiple TMA writers with unbalanced static acquire/commit sites}}
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_inconsistent_commit_fields_for_one_writer(
      %desc_a: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %desc_b: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %desc_c: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>,
      %b: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>,
      %c: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a, %b, %c {capacity = 2 : i32, pipe_name = "abc", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    tle.pipe.writer_acquire %a, %b, %c[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "abc", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %a_slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_a, %a_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a, %b, %c[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "abc", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    tle.pipe.writer_acquire %a, %b, %c[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "abc", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %b_slot = ttg.memdesc_index %b[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_b, %b_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a, %b, %c[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "abc", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    tle.pipe.writer_acquire %a, %b, %c[%c0, %false] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "abc", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %c_slot = ttg.memdesc_index %c[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_c, %c_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{uses a TMA writer task whose commit sites target different pipe field sets}}
    tle.pipe.writer_commit %a, %b, %c[%c0] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "abc", field_names = ["a", "b", "c"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_multi_writer_overlapping_fields(
      %desc_a: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %desc_b: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>,
      %b: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a, %b {capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %a_slot0 = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    %b_slot0 = ttg.memdesc_index %b[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_a, %a_slot0, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_b, %b_slot0, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %b_slot1 = ttg.memdesc_index %b[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_b, %b_slot1, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{uses multiple pure-TMA writers that target the same pipe field}}
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_multi_writer_incomplete_union(
      %desc_a: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %desc_b: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>,
      %b: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a, %b {capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %a_slot0 = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_a, %a_slot0, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>

    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %a_slot1 = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc_b, %a_slot1, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // expected-error @+1 {{uses multiple pure-TMA writers whose combined commits do not cover every pipe field}}
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "ab", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_multi_writer_overlapping_shared_allocation_subslices(
      %desc_a: !tt.tensordesc<tensor<32x64xf32, #nvmma>>,
      %desc_b: !tt.tensordesc<tensor<32x64xf32, #nvmma>>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %storage = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf32, #shared3, #smem, mutable>
    %a = ttg.memdesc_subslice %storage[0, 0, 0] : !ttg.memdesc<2x64x64xf32, #shared3, #smem, mutable> -> !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>
    %b = ttg.memdesc_subslice %storage[0, 0, 0] : !ttg.memdesc<2x64x64xf32, #shared3, #smem, mutable> -> !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>
    tle.pipe.create %a, %b {capacity = 2 : i32, pipe_name = "aliasing_subviews", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>

    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "aliasing_subviews", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>
    %a_slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable, 2x64x64>
    ttg.tma_copy %desc_a, %a_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable, 2x64x64>
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "aliasing_subviews", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>

    tle.pipe.writer_acquire %a, %b[%c0, %false] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "aliasing_subviews", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>
    %b_slot = ttg.memdesc_index %b[%c0] : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable, 2x64x64>
    ttg.tma_copy %desc_b, %b_slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable, 2x64x64>
    // expected-error @+1 {{fields share an allocation but are overlapping or not statically provable as disjoint subviews}}
    tle.pipe.writer_commit %a, %b[%c0] {async_task_id = array<i32: 1>, capacity = 2 : i32, pipe_name = "aliasing_subviews", field_names = ["a", "b"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>, !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>
    tt.return
  }
}
