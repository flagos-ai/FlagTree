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

// RUN: split-file %s %t
// RUN: triton-opt %t/valid.mlir -triton-tle-lower-pipe-to-nvws | FileCheck %s
// RUN: triton-opt %t/invalid.mlir -split-input-file -triton-tle-lower-pipe-to-nvws -verify-diagnostics

//--- valid.mlir
// Each logical 80x256 stage has 20 physical 16x64 tiles. Copies to
// representative tiles 0 and 5 must share the logical stage's TMA commit.
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#storage = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @single_tiled_stage
  tt.func @single_tiled_stage(%desc: !tt.tensordesc<tensor<16x64xf16, #nvmma>>) {
    // CHECK: %[[ZERO_STAGE:.*]] = arith.constant 0 : i32
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %false = arith.constant false
    %buf = ttg.local_alloc {alignment = 1024 : i32, tle.exact_smem_shape = array<i64: 1, 80, 256>, tle.smem_plan = {encoding = #nvmma, shape = array<i64: 80, 256>, tile = array<i64: 16, 64>, version = 1 : i32}} : () -> !ttg.memdesc<20x16x64xf16, #storage, #smem, mutable>
    // CHECK: %[[ZERO_TOKEN:.*]] = nvws.create_token {{.*}}loadType = 2 : i32, numBuffers = 1 : i32
    tle.pipe.create %buf {capacity = 1 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<20x16x64xf16, #storage, #smem, mutable>
    // CHECK: nvws.producer_acquire %[[ZERO_TOKEN]], %[[ZERO_STAGE]]
    tle.pipe.writer_acquire %buf[%c0, %false] {capacity = 1 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<20x16x64xf16, #storage, #smem, mutable>
    %tile0 = ttg.memdesc_index %buf[%c0] {tle.exact_smem_tile = 0 : i32} : !ttg.memdesc<20x16x64xf16, #storage, #smem, mutable> -> !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    // CHECK: ttg.tma_copy
    ttg.tma_copy %desc, %tile0, [%c0, %c0] : !tt.tensordesc<tensor<16x64xf16, #nvmma>>, !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    %tile5 = ttg.memdesc_index %buf[%c5] {tle.exact_smem_tile = 5 : i32} : !ttg.memdesc<20x16x64xf16, #storage, #smem, mutable> -> !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    // CHECK: ttg.tma_copy
    ttg.tma_copy %desc, %tile5, [%c0, %c0] : !tt.tensordesc<tensor<16x64xf16, #nvmma>>, !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    // CHECK: nvws.producer_commit %[[ZERO_TOKEN]], %[[ZERO_STAGE]] {{.*}}commitKind = 2 : i32
    tle.pipe.writer_commit %buf[%c0] {capacity = 1 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<20x16x64xf16, #storage, #smem, mutable>
    // CHECK-NOT: tle.pipe.
    tt.return
  }

  // Folding 1 * 20 + tile must not turn a tile index into a stage index.
  // CHECK-LABEL: tt.func @folded_tiled_stage
  tt.func @folded_tiled_stage(%desc: !tt.tensordesc<tensor<16x64xf16, #nvmma>>) {
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    // CHECK: %[[FOLDED_STAGE:.*]] = arith.constant 1 : i32
    %c1 = arith.constant 1 : i32
    %c20 = arith.constant 20 : i32
    %c25 = arith.constant 25 : i32
    %false = arith.constant false
    %buf = ttg.local_alloc {alignment = 1024 : i32, tle.exact_smem_shape = array<i64: 2, 80, 256>, tle.smem_plan = {encoding = #nvmma, shape = array<i64: 80, 256>, tile = array<i64: 16, 64>, version = 1 : i32}} : () -> !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    // CHECK: %[[FOLDED_TOKEN:.*]] = nvws.create_token {{.*}}loadType = 2 : i32, numBuffers = 2 : i32
    tle.pipe.create %buf {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    // CHECK: nvws.producer_acquire %[[FOLDED_TOKEN]], %[[FOLDED_STAGE]]
    tle.pipe.writer_acquire %buf[%c1, %false] {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    %tile0 = ttg.memdesc_index %buf[%c20] {tle.exact_smem_tile = 0 : i32} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable> -> !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    // CHECK: ttg.tma_copy
    ttg.tma_copy %desc, %tile0, [%c0, %c0] : !tt.tensordesc<tensor<16x64xf16, #nvmma>>, !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    %tile5 = ttg.memdesc_index %buf[%c25] {tle.exact_smem_tile = 5 : i32} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable> -> !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    // CHECK: ttg.tma_copy
    ttg.tma_copy %desc, %tile5, [%c0, %c0] : !tt.tensordesc<tensor<16x64xf16, #nvmma>>, !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    // CHECK: nvws.producer_commit %[[FOLDED_TOKEN]], %[[FOLDED_STAGE]] {{.*}}commitKind = 2 : i32
    tle.pipe.writer_commit %buf[%c1] {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    // CHECK-NOT: tle.pipe.
    tt.return
  }

  // CHECK-LABEL: tt.func @dynamic_tiled_stage
  // CHECK-SAME: %[[DYNAMIC_STAGE:arg[0-9]+]]: i32
  tt.func @dynamic_tiled_stage(%desc: !tt.tensordesc<tensor<16x64xf16, #nvmma>>, %stage: i32) {
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %c20 = arith.constant 20 : i32
    %false = arith.constant false
    %buf = ttg.local_alloc {alignment = 1024 : i32, tle.exact_smem_shape = array<i64: 2, 80, 256>, tle.smem_plan = {encoding = #nvmma, shape = array<i64: 80, 256>, tile = array<i64: 16, 64>, version = 1 : i32}} : () -> !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    // CHECK: %[[DYNAMIC_TOKEN:.*]] = nvws.create_token {{.*}}loadType = 2 : i32, numBuffers = 2 : i32
    tle.pipe.create %buf {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    // CHECK: nvws.producer_acquire %[[DYNAMIC_TOKEN]], %[[DYNAMIC_STAGE]]
    tle.pipe.writer_acquire %buf[%stage, %false] {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    %base = arith.muli %stage, %c20 : i32
    %offset = arith.addi %base, %c5 : i32
    %tile0 = ttg.memdesc_index %buf[%base] {tle.exact_smem_tile = 0 : i32} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable> -> !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    // CHECK: ttg.tma_copy
    ttg.tma_copy %desc, %tile0, [%c0, %c0] : !tt.tensordesc<tensor<16x64xf16, #nvmma>>, !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    %tile5 = ttg.memdesc_index %buf[%offset] {tle.exact_smem_tile = 5 : i32} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable> -> !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    // CHECK: ttg.tma_copy
    ttg.tma_copy %desc, %tile5, [%c0, %c0] : !tt.tensordesc<tensor<16x64xf16, #nvmma>>, !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    // CHECK: nvws.producer_commit %[[DYNAMIC_TOKEN]], %[[DYNAMIC_STAGE]] {{.*}}commitKind = 2 : i32
    tle.pipe.writer_commit %buf[%stage] {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    // CHECK-NOT: tle.pipe.
    tt.return
  }
}

//--- invalid.mlir
// Tile 25 belongs to stage 1, not the stage 0 being committed.
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#storage = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_tile_from_wrong_stage(%desc: !tt.tensordesc<tensor<16x64xf16, #nvmma>>) {
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %c25 = arith.constant 25 : i32
    %false = arith.constant false
    %buf = ttg.local_alloc {alignment = 1024 : i32, tle.exact_smem_shape = array<i64: 2, 80, 256>, tle.smem_plan = {encoding = #nvmma, shape = array<i64: 80, 256>, tile = array<i64: 16, 64>, version = 1 : i32}} : () -> !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    tle.pipe.create %buf {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    tle.pipe.writer_acquire %buf[%c0, %false] {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    %tile0 = ttg.memdesc_index %buf[%c0] {tle.exact_smem_tile = 0 : i32} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable> -> !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %tile0, [%c0, %c0] : !tt.tensordesc<tensor<16x64xf16, #nvmma>>, !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    %tile5 = ttg.memdesc_index %buf[%c25] {tle.exact_smem_tile = 5 : i32} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable> -> !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %tile5, [%c0, %c0] : !tt.tensordesc<tensor<16x64xf16, #nvmma>>, !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    // expected-error @+1 {{has an unrelated ttg.tma_copy between pipe payload TMA copies and commit}}
    tle.pipe.writer_commit %buf[%c0] {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    tt.return
  }
}

// -----

// A matching logical stage does not make another allocation this pipe's field.
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#storage = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_tile_from_other_field(%desc: !tt.tensordesc<tensor<16x64xf16, #nvmma>>) {
    %c0 = arith.constant 0 : i32
    %c5 = arith.constant 5 : i32
    %c25 = arith.constant 25 : i32
    %false = arith.constant false
    %buf = ttg.local_alloc {alignment = 1024 : i32, tle.exact_smem_shape = array<i64: 2, 80, 256>, tle.smem_plan = {encoding = #nvmma, shape = array<i64: 80, 256>, tile = array<i64: 16, 64>, version = 1 : i32}} : () -> !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    %other = ttg.local_alloc {alignment = 1024 : i32, tle.exact_smem_shape = array<i64: 2, 80, 256>, tle.smem_plan = {encoding = #nvmma, shape = array<i64: 80, 256>, tile = array<i64: 16, 64>, version = 1 : i32}} : () -> !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    tle.pipe.create %buf {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    tle.pipe.writer_acquire %buf[%c0, %false] {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    %tile0 = ttg.memdesc_index %buf[%c0] {tle.exact_smem_tile = 0 : i32} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable> -> !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %tile0, [%c0, %c0] : !tt.tensordesc<tensor<16x64xf16, #nvmma>>, !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    %tile5 = ttg.memdesc_index %other[%c5] {tle.exact_smem_tile = 5 : i32} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable> -> !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %tile5, [%c0, %c0] : !tt.tensordesc<tensor<16x64xf16, #nvmma>>, !ttg.memdesc<16x64xf16, #nvmma, #smem, mutable>
    // expected-error @+1 {{has an unrelated ttg.tma_copy between pipe payload TMA copies and commit}}
    tle.pipe.writer_commit %buf[%c0] {capacity = 2 : i32, pipe_name = "k", field_names = ["k"], scope = "cta", tiled_smem_fields = array<i32: 0>} : !ttg.memdesc<40x16x64xf16, #storage, #smem, mutable>
    tt.return
  }
}
