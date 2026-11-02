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

// RUN: triton-opt %s -split-input-file -tritongpu-pipeline=num-stages=3 | FileCheck %s --check-prefixes=COMMON,OFF
// RUN: triton-opt %s -split-input-file -tritongpu-pipeline="num-stages=3 enable-pipeline-enhance=true" | FileCheck %s --check-prefixes=COMMON,ON

// The same block is scaled in registers for the dot's A operand and read
// transposed from shared memory for its B operand. The wgmma keeps reading
// the B buffer asynchronously while the prefetch of the next block is issued,
// so the ring needs one slot more than the load distance. Without the fix the
// ring has only two slots and the prefetch overwrites the buffer in use.

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // COMMON-LABEL: @load_feeds_registers_and_wgmma_shared_operand
  // OFF: ttg.local_alloc : () -> !ttg.memdesc<2x64x32xf16
  // ON: ttg.local_alloc : () -> !ttg.memdesc<3x64x32xf16
  // COMMON: scf.for
  // COMMON: ttng.warp_group_dot
  // COMMON: ttg.async_copy_global_to_local
  tt.func public @load_feeds_registers_and_wgmma_shared_operand(
      %a_base: !tt.ptr<f16> {tt.divisibility = 16 : i32},
      %scale: tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) -> tensor<64x64xf32, #mma> {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c32_i32 = arith.constant 32 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %cst_row = arith.constant dense<128> : tensor<64x1xi32, #blocked>
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows2 = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %row_off = arith.muli %rows2, %cst_row : tensor<64x1xi32, #blocked>
    %row_off2 = tt.broadcast %row_off : tensor<64x1xi32, #blocked> -> tensor<64x32xi32, #blocked>
    %cols = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %base = tt.splat %a_base : !tt.ptr<f16> -> tensor<64x32x!tt.ptr<f16>, #blocked>
    %res = scf.for %i = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%acc = %cst) -> (tensor<64x64xf32, #mma>) : i32 {
      %k_off = arith.muli %i, %c32_i32 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32
      %k_off1 = tt.splat %k_off {loop.cluster = 2 : i32, loop.stage = 0 : i32} : i32 -> tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %col = arith.addi %k_off1, %cols {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %col2 = tt.expand_dims %col {axis = 0 : i32, loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x32xi32, #blocked>
      %col3 = tt.broadcast %col2 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<1x32xi32, #blocked> -> tensor<64x32xi32, #blocked>
      %off = arith.addi %row_off2, %col3 {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x32xi32, #blocked>
      %ptr = tt.addptr %base, %off {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #blocked>, tensor<64x32xi32, #blocked>
      %k = tt.load %ptr {loop.cluster = 2 : i32, loop.stage = 0 : i32} : tensor<64x32x!tt.ptr<f16>, #blocked>
      %k_reg = ttg.convert_layout %k {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x32xf16, #blocked> -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %k_scaled = arith.mulf %k_reg, %scale {loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      %k_smem = ttg.local_alloc %k {loop.cluster = 0 : i32, loop.stage = 2 : i32} : (tensor<64x32xf16, #blocked>) -> !ttg.memdesc<64x32xf16, #shared, #smem>
      %k_trans = ttg.memdesc_trans %k_smem {loop.cluster = 0 : i32, loop.stage = 2 : i32, order = array<i32: 1, 0>} : !ttg.memdesc<64x32xf16, #shared, #smem> -> !ttg.memdesc<32x64xf16, #shared1, #smem>
      %d = ttng.warp_group_dot %k_scaled, %k_trans, %acc {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 2 : i32} : tensor<64x32xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<32x64xf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      scf.yield %d : tensor<64x64xf32, #mma>
    } {tt.scheduled_max_stage = 2 : i32}
    tt.return %res : tensor<64x64xf32, #mma>
  }
}
