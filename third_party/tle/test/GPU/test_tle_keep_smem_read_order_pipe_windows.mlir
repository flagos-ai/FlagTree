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
// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-loads=keep-smem-read-order=true --tritongpu-accelerate-matmul | FileCheck %s --check-prefix=STAGE

// Which pipe operations keep a load of a pipe slot in place (keep-smem-read-order).
//
// A reader_wait, a writer_acquire and a writer_commit move no data, so they do
// not stop the load from being re-issued later. A reader_release gives the slot
// back to its writer, which may overwrite it from then on: the load stays
// before it, unless the field the release names is a window of the buffer that
// is provably apart from the window the load reads. Here the two fields are the
// left and the right half of one allocation, declared as such (subslices of the
// buffer). When the fields are the whole buffer and only the writers fill the
// halves, test_tle_keep_smem_read_order_pipe_writers.mlir has the decision.
//
// The held load is also what costs shared memory: a load kept in registers
// across the release is staged by accelerate-matmul in a second buffer for the
// dot (a 64x256 bf16 tile, 32768 bytes). The STAGE run pins that the load of the
// other half is not held and the dot reads the slot, and that the load of the
// released half is held and staged.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked_col = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared_slot = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The release is of the left half and the load reads the right half: the load
  // is re-issued after the release, at its conversion.
  // COMMON-LABEL: tt.func @release_of_the_other_half
  // STAGE-LABEL: tt.func @release_of_the_other_half
  // STAGE-NOT: ttg.local_alloc %{{.*}} : (tensor<64x256xbf16
  // STAGE: ttng.warp_group_dot
  tt.func @release_of_the_other_half(%a: tensor<64x64xbf16, #blocked>, %acc: tensor<64x256xf32, #blocked>, %out: tensor<64x256x!tt.ptr<f32>, #blocked>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
    %left = ttg.memdesc_subslice %buf[0, 0, 0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>
    %right = ttg.memdesc_subslice %buf[0, 0, 256] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>
    ttg.warp_specialize(%left, %right, %a, %acc, %out) attributes {requestedRegisters = array<i32: 216>}
    default {
      ttg.warp_yield
    }
    partition0(%l: !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>, %r: !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %slot = ttg.memdesc_index %r[%c0] : !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 1x64x512>
      %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_col}>>
      %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_col}>> -> tensor<64x1xi32, #blocked_col>
      %rowsb = tt.broadcast %rows2 : tensor<64x1xi32, #blocked_col> -> tensor<64x256xi32, #blocked_col>
      %cols = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked_col}>>
      %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked_col}>> -> tensor<1x256xi32, #blocked_col>
      %colsb = tt.broadcast %cols2 : tensor<1x256xi32, #blocked_col> -> tensor<64x256xi32, #blocked_col>
      %ptr = "tle.local_pointers"(%slot, %rowsb, %colsb) : (!ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 1x64x512>, tensor<64x256xi32, #blocked_col>, tensor<64x256xi32, #blocked_col>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      // The load is re-issued after the release with the option, as it is
      // without it.
      // COMMON: tle.pipe.reader_release
      // COMMON: ttg.local_load
      %b = tt.load %ptr : tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      tle.pipe.reader_release %l[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "l", reader_name = "qk", scope = "cta"} : !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>
      %b_dot = ttg.convert_layout %b : tensor<64x256xbf16, #blocked_col> -> tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %a_dot = ttg.convert_layout %arg1 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %r2 = tt.dot %a_dot, %b_dot, %arg2, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x256xf32, #blocked>
      tt.store %arg3, %r2 : tensor<64x256x!tt.ptr<f32>, #blocked>
      ttg.warp_return
    } : (!ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>, !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>, tensor<64x64xbf16, #blocked>, tensor<64x256xf32, #blocked>, tensor<64x256x!tt.ptr<f32>, #blocked>) -> ()
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
  // The release is of the half the load reads: the writer may overwrite it, so
  // the load stays before the release.
  // COMMON-LABEL: tt.func @release_of_the_same_half
  // STAGE-LABEL: tt.func @release_of_the_same_half
  // STAGE: ttg.local_alloc %{{.*}} : (tensor<64x256xbf16
  // STAGE: ttng.warp_group_dot
  tt.func @release_of_the_same_half(%a: tensor<64x64xbf16, #blocked>, %acc: tensor<64x256xf32, #blocked>, %out: tensor<64x256x!tt.ptr<f32>, #blocked>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable>
    %left = ttg.memdesc_subslice %buf[0, 0, 0] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>
    %right = ttg.memdesc_subslice %buf[0, 0, 256] : !ttg.memdesc<1x64x512xbf16, #shared, #smem, mutable> -> !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>
    ttg.warp_specialize(%left, %right, %a, %acc, %out) attributes {requestedRegisters = array<i32: 216>}
    default {
      ttg.warp_yield
    }
    partition0(%l: !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>, %r: !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %slot = ttg.memdesc_index %r[%c0] : !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 1x64x512>
      %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_col}>>
      %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked_col}>> -> tensor<64x1xi32, #blocked_col>
      %rowsb = tt.broadcast %rows2 : tensor<64x1xi32, #blocked_col> -> tensor<64x256xi32, #blocked_col>
      %cols = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked_col}>>
      %cols2 = tt.expand_dims %cols {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked_col}>> -> tensor<1x256xi32, #blocked_col>
      %colsb = tt.broadcast %cols2 : tensor<1x256xi32, #blocked_col> -> tensor<64x256xi32, #blocked_col>
      %ptr = "tle.local_pointers"(%slot, %rowsb, %colsb) : (!ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable, 1x64x512>, tensor<64x256xi32, #blocked_col>, tensor<64x256xi32, #blocked_col>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      // With the option the load stays before the release; without it the load
      // is re-issued after the release.
      // ON: ttg.local_load
      // ON-NEXT: tle.pipe.reader_release
      // OFF: tle.pipe.reader_release
      // OFF: ttg.local_load
      %b = tt.load %ptr : tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      tle.pipe.reader_release %r[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "r", reader_name = "qk", scope = "cta"} : !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>
      %b_dot = ttg.convert_layout %b : tensor<64x256xbf16, #blocked_col> -> tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      %a_dot = ttg.convert_layout %arg1 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %r2 = tt.dot %a_dot, %b_dot, %arg2, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x256xf32, #blocked>
      tt.store %arg3, %r2 : tensor<64x256x!tt.ptr<f32>, #blocked>
      ttg.warp_return
    } : (!ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>, !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable, 1x64x512>, tensor<64x64xbf16, #blocked>, tensor<64x256xf32, #blocked>, tensor<64x256x!tt.ptr<f32>, #blocked>) -> ()
    tt.return
  }
}
