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
// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-loads=keep-smem-read-order=true --tritongpu-accelerate-matmul | FileCheck %s --check-prefixes=COMMON,ON
// RUN: triton-opt %s -split-input-file --triton-tle-optimize-local-pointer-loads --tritongpu-accelerate-matmul | FileCheck %s --check-prefixes=COMMON,OFF

// The shared-memory cost of keep-smem-read-order in
// _tle_flashmla_prefill_consumer1 (python/tutorials/tle/deepseek_v32/
// 02-sparse-mla.py): inside a warp_specialize partition, a local pointer load of
// a pipe slot, a tle.pipe.reader_release between the load and the dot, and a
// dot whose B operand is the loaded value. The buffer is a warp_specialize
// capture, so in the partition it is a block argument.
//
// Here the pipe of the release is over the whole buffer the load reads, so the
// release may hand the bytes back to the writer and the load stays where it is.
// accelerate-matmul then stages the loaded register value in a second shared
// memory buffer: a ttg.local_alloc of tensor<64x256xbf16>, 32768 bytes in the
// kernel (263920 bytes of shared memory against 231824 without the option; sm_90
// allows 232448). Without the option the load is re-issued at the dot and the dot
// reads the pipe slot. The test pins that cost for this case, which is the
// intended behaviour: the release is a write of the buffer.
//
// The cost goes away when the pipes are declared over windows of the buffer that
// do not overlap: test_tle_keep_smem_read_order_pipe_windows.mlir holds the same
// kernel shape with the release of the other half, where the load is not held and
// nothing is staged. The tutorial's sK1_l and sK1_r are pipes over one buffer;
// declaring them over its two column halves (sK1_smem.subslice) gives the kernel
// that form, but the PPU build rejects a memdesc_index of a subview and the
// tutorial builds there too. It reads the slot again before the dot instead, as
// consumer0 does.
//
// The register layout of the load has order [0, 1], as the kernel's sK copy has
// when it is an async copy. With order [1, 0] the staged encoding equals the
// buffer's own, accelerate-matmul uses the buffer for the dot and stages
// nothing, with or without the option.

#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#blocked_col = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16, rank = 3}>
#shared_slot = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: tt.func @pipe_release_of_same_buffer_between_load_and_dot
  tt.func @pipe_release_of_same_buffer_between_load_and_dot(%a: tensor<64x64xbf16, #blocked>, %acc: tensor<64x256xf32, #blocked>, %out: tensor<64x256x!tt.ptr<f32>, #blocked>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable>
    // COMMON: ttg.warp_specialize
    ttg.warp_specialize(%buf, %a, %acc, %out) attributes {requestedRegisters = array<i32: 216>}
    default {
      ttg.warp_yield
    }
    partition0(%arg0: !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable>, %arg1: tensor<64x64xbf16, #blocked>, %arg2: tensor<64x256xf32, #blocked>, %arg3: tensor<64x256x!tt.ptr<f32>, #blocked>) num_warps(4) {
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      // COMMON: tle.pipe.reader_wait
      %wait = tle.pipe.reader_wait %arg0[%c0, %false] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_r", scope = "cta"} : !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable>
      // COMMON: %[[SLOT:.*]] = ttg.memdesc_index
      %slot = ttg.memdesc_index %arg0[%c0] : !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable>
      %ptr = "tle.local_pointers"(%slot) : (!ttg.memdesc<64x256xbf16, #shared_slot, #smem, mutable>) -> tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      // ON: %[[HELD:.*]] = ttg.local_load
      // ON: %[[STAGED:.*]] = ttg.local_alloc %[[HELD]] : (tensor<64x256xbf16, #{{.*}}>) -> !ttg.memdesc<64x256xbf16, #{{.*}}, #smem>
      // OFF-NOT: ttg.local_load
      // OFF-NOT: ttg.local_alloc %{{.*}} : (tensor<64x256xbf16
      %b = tt.load %ptr : tensor<64x256x!tt.ptr<bf16, 3>, #blocked_col>
      // COMMON: tle.pipe.reader_release
      tle.pipe.reader_release %arg0[%c0] {capacity = 1 : i32, field_names = ["sK"], pipe_name = "sK_l", reader_name = "qk", scope = "cta"} : !ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable>
      %a_dot = ttg.convert_layout %arg1 : tensor<64x64xbf16, #blocked> -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
      %b_dot = ttg.convert_layout %b : tensor<64x256xbf16, #blocked_col> -> tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
      // ON: ttng.warp_group_dot %{{.*}}, %[[STAGED]],
      // OFF: ttng.warp_group_dot %{{.*}}, %[[SLOT]],
      %r = tt.dot %a_dot, %b_dot, %arg2, inputPrecision = tf32 : tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x256xf32, #blocked>
      tt.store %arg3, %r : tensor<64x256x!tt.ptr<f32>, #blocked>
      ttg.warp_return
    } : (!ttg.memdesc<1x64x256xbf16, #shared, #smem, mutable>, tensor<64x64xbf16, #blocked>, tensor<64x256xf32, #blocked>, tensor<64x256x!tt.ptr<f32>, #blocked>) -> ()
    tt.return
  }
}
