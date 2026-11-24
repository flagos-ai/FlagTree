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

// REQUIRES: flagtree-keep-smem-read-order
// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-loads=keep-smem-read-order=true | FileCheck %s --check-prefixes=COMMON,ON
// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-loads | FileCheck %s --check-prefixes=COMMON,OFF
// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-loads=keep-smem-read-order=true --tritongpu-accelerate-matmul | FileCheck %s --check-prefixes=COMMON,STAGE

// A pipe names its fields, and a reader_release hands a slot of each of them
// back to the pipe's writer. What the writer may then overwrite is what it
// writes between its writer_acquire and its writer_commit, not what the pipe
// names: the sparse-MLA tutorial (python/tutorials/tle/deepseek_v32/
// 02-sparse-mla.py) declares both sK1 pipes over the whole of one buffer, and
// the writer of sK1_l fills the left half of it while the writer of sK1_r fills
// the right half. The consumer loads the right half, releases sK1_l, and then
// runs the dot on what it loaded. The release gives back the left half, so the
// load of the right half need not be held before it; held, it is a register
// value that accelerate-matmul stages in a second shared memory buffer for the
// dot (32768 bytes, which made the kernel need 263920 bytes of shared memory
// against the 232448 of sm_90).
//
// The tests below fix what the release is a write of: what the writer of its
// pipe fills, and the whole of the field when that is not known. Without the
// option the load is re-issued after the release, as before.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked_col = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared_slot = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The writers of the two pipes fill the left and the right half; both pipes
  // name the whole buffer. The consumer loads the right half and releases the
  // pipe whose writer fills the left half: the load is not held, and the dot
  // reads the slot.
  // COMMON-LABEL: tt.func @writers_of_two_halves
  tt.func @writers_of_two_halves(%a: tensor<64x64xbf16, #blocked>, %acc: tensor<64x256xf32, #blocked>, %out: tensor<64x256x!tt.ptr<f32>, #blocked>, %src: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
    ttg.warp_specialize(%buf, %a, %acc, %out, %src) attributes {requestedRegisters = array<i32: 216, 216>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>, %arg4: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      %wait = tle.pipe.reader_wait %arg0[%c0, %false] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %slot = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %right = ttg.memdesc_subslice %slot[0, 256] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %ptr = "tle.local_pointers"(%right) : (!ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      // The load is re-issued after the release, with the option as without it:
      // the release gives back the left half, which the load does not read. With
      // accelerate-matmul behind it, the dot reads the slot and nothing is
      // staged.
      // COMMON: tle.pipe.reader_release
      // ON: ttg.local_load
      // OFF: ttg.local_load
      // STAGE-NOT: ttg.local_alloc %{{.*}} : (tensor<64x256xbf16
      // STAGE: ttng.warp_group_dot
      %b = tt.load %ptr : tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      tle.pipe.reader_release %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", reader_name = "qk", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %a_dot = ttg.convert_layout %arg1 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b_dot = ttg.convert_layout %b : tensor<64x256xbf16, #blocked_col> -> tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %r = tt.dot %a_dot, %b_dot, %arg2, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x256xf32, #blocked>
      tt.store %arg3, %r : tensor<64x256x!tt.ptr<f32>, #blocked>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>, %arg4: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %true = arith.constant true
      // The writer of sK_l fills the left half, the writer of sK_r the right.
      tle.pipe.writer_acquire %arg0[%c0, %true] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %wslot_l = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %left = ttg.memdesc_subslice %wslot_l[0, 0] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %lt = ttg.async_copy_global_to_local %arg4, %left : tensor<64x256x!tt.ptr<bf16>, #blocked_col> -> <64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      tle.pipe.writer_commit %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      tle.pipe.writer_acquire %arg0[%c0, %true] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %wslot_r = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %rightw = ttg.memdesc_subslice %wslot_r[0, 256] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %rt = ttg.async_copy_global_to_local %arg4, %rightw : tensor<64x256x!tt.ptr<bf16>, #blocked_col> -> <64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      tle.pipe.writer_commit %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, tensor<64x64xbf16, #blocked>, tensor<64x256xf32, #blocked>, tensor<64x256x!tt.ptr<f32>, #blocked>, tensor<64x256x!tt.ptr<bf16>, #blocked_col>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked_col = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared_slot = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The release is of the pipe whose writer fills the half the load reads: the
  // writer may overwrite it from then on, so the load stays before the release.
  // COMMON-LABEL: tt.func @release_of_the_writer_of_the_same_half
  tt.func @release_of_the_writer_of_the_same_half(%a: tensor<64x64xbf16, #blocked>, %acc: tensor<64x256xf32, #blocked>, %out: tensor<64x256x!tt.ptr<f32>, #blocked>, %src: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
    ttg.warp_specialize(%buf, %a, %acc, %out, %src) attributes {requestedRegisters = array<i32: 216, 216>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>, %arg4: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      %wait = tle.pipe.reader_wait %arg0[%c0, %false] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %slot = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %view = ttg.memdesc_subslice %slot[0, 256] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %ptr = "tle.local_pointers"(%view) : (!ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      // ON: ttg.memdesc_subslice
      // ON: ttg.local_load
      // ON-NEXT: tle.pipe.reader_release
      // OFF: tle.pipe.reader_release
      // OFF: ttg.local_load
      %b = tt.load %ptr : tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      tle.pipe.reader_release %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", reader_name = "qk", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %a_dot = ttg.convert_layout %arg1 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b_dot = ttg.convert_layout %b : tensor<64x256xbf16, #blocked_col> -> tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %r = tt.dot %a_dot, %b_dot, %arg2, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x256xf32, #blocked>
      tt.store %arg3, %r : tensor<64x256x!tt.ptr<f32>, #blocked>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>, %arg4: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %true = arith.constant true
      tle.pipe.writer_acquire %arg0[%c0, %true] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %ws_l = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %wv_l = ttg.memdesc_subslice %ws_l[0, 0] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %wc_l = ttg.async_copy_global_to_local %arg4, %wv_l : tensor<64x256x!tt.ptr<bf16>, #blocked_col> -> <64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      tle.pipe.writer_commit %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      tle.pipe.writer_acquire %arg0[%c0, %true] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %ws_r = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %wv_r = ttg.memdesc_subslice %ws_r[0, 256] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %wc_r = ttg.async_copy_global_to_local %arg4, %wv_r : tensor<64x256x!tt.ptr<bf16>, #blocked_col> -> <64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      tle.pipe.writer_commit %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>

      ttg.warp_return
    } : (!ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, tensor<64x64xbf16, #blocked>, tensor<64x256xf32, #blocked>, tensor<64x256x!tt.ptr<f32>, #blocked>, tensor<64x256x!tt.ptr<bf16>, #blocked_col>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked_col = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared_slot = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The release is of a pipe whose writer fills both halves: it may overwrite
  // the half the load reads, so the load stays before the release.
  // COMMON-LABEL: tt.func @writer_of_the_released_pipe_fills_the_whole_buffer
  tt.func @writer_of_the_released_pipe_fills_the_whole_buffer(%a: tensor<64x64xbf16, #blocked>, %acc: tensor<64x256xf32, #blocked>, %out: tensor<64x256x!tt.ptr<f32>, #blocked>, %src: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
    ttg.warp_specialize(%buf, %a, %acc, %out, %src) attributes {requestedRegisters = array<i32: 216, 216>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>, %arg4: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      %wait = tle.pipe.reader_wait %arg0[%c0, %false] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %slot = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %view = ttg.memdesc_subslice %slot[0, 256] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %ptr = "tle.local_pointers"(%view) : (!ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      // ON: ttg.memdesc_subslice
      // ON: ttg.local_load
      // ON-NEXT: tle.pipe.reader_release
      // OFF: tle.pipe.reader_release
      // OFF: ttg.local_load
      %b = tt.load %ptr : tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      tle.pipe.reader_release %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", reader_name = "qk", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %a_dot = ttg.convert_layout %arg1 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b_dot = ttg.convert_layout %b : tensor<64x256xbf16, #blocked_col> -> tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %r = tt.dot %a_dot, %b_dot, %arg2, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x256xf32, #blocked>
      tt.store %arg3, %r : tensor<64x256x!tt.ptr<f32>, #blocked>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>, %arg4: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %true = arith.constant true
      tle.pipe.writer_acquire %arg0[%c0, %true] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %ws_l = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %wv_l = ttg.memdesc_subslice %ws_l[0, 0] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %wc_l = ttg.async_copy_global_to_local %arg4, %wv_l : tensor<64x256x!tt.ptr<bf16>, #blocked_col> -> <64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %wv2_l = ttg.memdesc_subslice %ws_l[0, 256] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %wc2_l = ttg.async_copy_global_to_local %arg4, %wv2_l : tensor<64x256x!tt.ptr<bf16>, #blocked_col> -> <64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      tle.pipe.writer_commit %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      tle.pipe.writer_acquire %arg0[%c0, %true] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %ws_r = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %wv_r = ttg.memdesc_subslice %ws_r[0, 256] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %wc_r = ttg.async_copy_global_to_local %arg4, %wv_r : tensor<64x256x!tt.ptr<bf16>, #blocked_col> -> <64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      tle.pipe.writer_commit %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>

      ttg.warp_return
    } : (!ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, tensor<64x64xbf16, #blocked>, tensor<64x256xf32, #blocked>, tensor<64x256x!tt.ptr<f32>, #blocked>, tensor<64x256x!tt.ptr<bf16>, #blocked_col>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked_col = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared_slot = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // An op in the writer region that declares no effects (gpu.barrier) may write
  // anything: what the writer fills is not known, and the load stays before the
  // release.
  // COMMON-LABEL: tt.func @writer_with_effects_not_known
  tt.func @writer_with_effects_not_known(%a: tensor<64x64xbf16, #blocked>, %acc: tensor<64x256xf32, #blocked>, %out: tensor<64x256x!tt.ptr<f32>, #blocked>, %src: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
    ttg.warp_specialize(%buf, %a, %acc, %out, %src) attributes {requestedRegisters = array<i32: 216, 216>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>, %arg4: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      %wait = tle.pipe.reader_wait %arg0[%c0, %false] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %slot = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %view = ttg.memdesc_subslice %slot[0, 256] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %ptr = "tle.local_pointers"(%view) : (!ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      // ON: ttg.memdesc_subslice
      // ON: ttg.local_load
      // ON-NEXT: tle.pipe.reader_release
      // OFF: tle.pipe.reader_release
      // OFF: ttg.local_load
      %b = tt.load %ptr : tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      tle.pipe.reader_release %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", reader_name = "qk", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %a_dot = ttg.convert_layout %arg1 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b_dot = ttg.convert_layout %b : tensor<64x256xbf16, #blocked_col> -> tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %r = tt.dot %a_dot, %b_dot, %arg2, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x256xf32, #blocked>
      tt.store %arg3, %r : tensor<64x256x!tt.ptr<f32>, #blocked>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>, %arg4: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %true = arith.constant true
      tle.pipe.writer_acquire %arg0[%c0, %true] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %ws_l = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %wv_l = ttg.memdesc_subslice %ws_l[0, 0] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %wc_l = ttg.async_copy_global_to_local %arg4, %wv_l : tensor<64x256x!tt.ptr<bf16>, #blocked_col> -> <64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      gpu.barrier
      tle.pipe.writer_commit %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      tle.pipe.writer_acquire %arg0[%c0, %true] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %ws_r = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %wv_r = ttg.memdesc_subslice %ws_r[0, 256] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %wc_r = ttg.async_copy_global_to_local %arg4, %wv_r : tensor<64x256x!tt.ptr<bf16>, #blocked_col> -> <64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      tle.pipe.writer_commit %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>

      ttg.warp_return
    } : (!ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, tensor<64x64xbf16, #blocked>, tensor<64x256xf32, #blocked>, tensor<64x256x!tt.ptr<f32>, #blocked>, tensor<64x256x!tt.ptr<bf16>, #blocked_col>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked_col = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared_slot = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // No writer region of the released pipe is in the module: nothing is known of
  // what its writer fills, and the load stays before the release.
  // COMMON-LABEL: tt.func @pipe_without_a_writer_in_the_module
  tt.func @pipe_without_a_writer_in_the_module(%a: tensor<64x64xbf16, #blocked>, %acc: tensor<64x256xf32, #blocked>, %out: tensor<64x256x!tt.ptr<f32>, #blocked>, %src: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
    ttg.warp_specialize(%buf, %a, %acc, %out, %src) attributes {requestedRegisters = array<i32: 216, 216>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>, %arg4: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      %wait = tle.pipe.reader_wait %arg0[%c0, %false] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %slot = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %view = ttg.memdesc_subslice %slot[0, 256] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %ptr = "tle.local_pointers"(%view) : (!ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      // ON: ttg.memdesc_subslice
      // ON: ttg.local_load
      // ON-NEXT: tle.pipe.reader_release
      // OFF: tle.pipe.reader_release
      // OFF: ttg.local_load
      %b = tt.load %ptr : tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      tle.pipe.reader_release %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", reader_name = "qk", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %a_dot = ttg.convert_layout %arg1 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b_dot = ttg.convert_layout %b : tensor<64x256xbf16, #blocked_col> -> tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %r = tt.dot %a_dot, %b_dot, %arg2, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x256xf32, #blocked>
      tt.store %arg3, %r : tensor<64x256x!tt.ptr<f32>, #blocked>
      ttg.warp_return
    }
    partition1(%arg0: !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>, %arg4: tensor<64x256x!tt.ptr<bf16>, #blocked_col>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %true = arith.constant true
      tle.pipe.writer_acquire %arg0[%c0, %true] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_x", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %ws_x = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %wv_x = ttg.memdesc_subslice %ws_x[0, 0] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %wc_x = ttg.async_copy_global_to_local %arg4, %wv_x : tensor<64x256x!tt.ptr<bf16>, #blocked_col> -> <64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      tle.pipe.writer_commit %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_x", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      tle.pipe.writer_acquire %arg0[%c0, %true] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
      %ws_r = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable>
      %wv_r = ttg.memdesc_subslice %ws_r[0, 256] : !ttg.memdesc<64x512xbf16, #shared_slot, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      %wc_r = ttg.async_copy_global_to_local %arg4, %wv_r : tensor<64x256x!tt.ptr<bf16>, #blocked_col> -> <64x256xbf16, #shared_slot, #smem, mutable, 64x512>
      tle.pipe.writer_commit %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>

      ttg.warp_return
    } : (!ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>, tensor<64x64xbf16, #blocked>, tensor<64x256xf32, #blocked>, tensor<64x256x!tt.ptr<f32>, #blocked>, tensor<64x256x!tt.ptr<bf16>, #blocked_col>) -> ()
    tt.return
  }
}
