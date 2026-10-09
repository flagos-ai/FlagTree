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

// RUN: triton-opt %s -split-input-file -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=120 ptx-version=88})' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @extract_tile_smem
  // CHECK: tle.extract_tile
  // CHECK-SAME: allocation.offset =
  tt.func @extract_tile_smem(%src: tensor<32x32xf32, #blocked>, %idx: i32) -> tensor<16x16xf32, #blocked> {
    %tile = tle.extract_tile %src[%idx] {tile_shape = array<i64: 16, 16>} : tensor<32x32xf32, #blocked>, i32 -> tensor<16x16xf32, #blocked>
    tt.return %tile : tensor<16x16xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @insert_tile_smem
  // CHECK: tle.insert_tile
  // CHECK-SAME: allocation.offset =
  tt.func @insert_tile_smem(%src: tensor<32x32xf32, #blocked>, %tile: tensor<16x16xf32, #blocked>, %idx: i32) -> tensor<32x32xf32, #blocked> {
    %result = tle.insert_tile %src[%idx] = %tile {tile_shape = array<i64: 16, 16>} : tensor<32x32xf32, #blocked>, i32, tensor<16x16xf32, #blocked> -> tensor<32x32xf32, #blocked>
    tt.return %result : tensor<32x32xf32, #blocked>
  }
}

// -----

// KDA's row layout places all 16 rows in each lane's registers. Both ends of
// the source tensor must use the same register-only path as LLVM lowering.
#row = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK: module attributes {{.*}}ttg.shared = 0 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @extract_static_rows_no_smem
  // CHECK: tle.extract_tile {{.*}} {tile_shape = array<i64: 1, 128>}
  // CHECK: tle.extract_tile {{.*}} {tile_shape = array<i64: 1, 128>}
  // CHECK: tt.return
  tt.func @extract_static_rows_no_smem(%src: tensor<16x128xf32, #row>) -> tensor<1x128xf32, #row> {
    %c0 = arith.constant 0 : i32
    %c15 = arith.constant 15 : i32
    %first = tle.extract_tile %src[%c0] {tile_shape = array<i64: 1, 128>} : tensor<16x128xf32, #row>, i32 -> tensor<1x128xf32, #row>
    %last = tle.extract_tile %src[%c15] {tile_shape = array<i64: 1, 128>} : tensor<16x128xf32, #row>, i32 -> tensor<1x128xf32, #row>
    %sum = arith.addf %first, %last : tensor<1x128xf32, #row>
    tt.return %sum : tensor<1x128xf32, #row>
  }
}

// -----

// A dynamic index must retain the existing shared-memory implementation.
#row = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK: module attributes {{.*}}ttg.shared = 512 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @extract_dynamic_row_smem
  // CHECK: tle.extract_tile
  // CHECK-SAME: allocation.offset = 0 : i32
  tt.func @extract_dynamic_row_smem(%src: tensor<16x128xf32, #row>, %idx: i32) -> tensor<1x128xf32, #row> {
    %tile = tle.extract_tile %src[%idx] {tile_shape = array<i64: 1, 128>} : tensor<16x128xf32, #row>, i32 -> tensor<1x128xf32, #row>
    tt.return %tile : tensor<1x128xf32, #row>
  }
}

// -----

// Static alone is insufficient: a one-row tile is smaller than the two-row
// CTA tile, so extracting row 15 changes which lane owns each value.
#rows = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK: module attributes {{.*}}ttg.shared = 512 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @extract_static_unaligned_row_smem
  // CHECK: tle.extract_tile
  // CHECK-SAME: allocation.offset = 0 : i32
  tt.func @extract_static_unaligned_row_smem(%src: tensor<16x128xf32, #rows>) -> tensor<1x128xf32, #rows> {
    %c15 = arith.constant 15 : i32
    %tile = tle.extract_tile %src[%c15] {tile_shape = array<i64: 1, 128>} : tensor<16x128xf32, #rows>, i32 -> tensor<1x128xf32, #rows>
    tt.return %tile : tensor<1x128xf32, #rows>
  }
}

// -----

// Equal warp counts do not imply equal lane ownership. Keep scratch when an
// explicit result layout redistributes values between lanes.
#row = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#rows = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [1, 4], order = [1, 0]}>

// CHECK: module attributes {{.*}}ttg.shared = 512 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @extract_static_changed_lane_layout_smem
  // CHECK: tle.extract_tile
  // CHECK-SAME: allocation.offset = 0 : i32
  tt.func @extract_static_changed_lane_layout_smem(%src: tensor<16x128xf32, #row>) -> tensor<1x128xf32, #rows> {
    %c15 = arith.constant 15 : i32
    %tile = tle.extract_tile %src[%c15] {tile_shape = array<i64: 1, 128>} : tensor<16x128xf32, #row>, i32 -> tensor<1x128xf32, #rows>
    tt.return %tile : tensor<1x128xf32, #rows>
  }
}
