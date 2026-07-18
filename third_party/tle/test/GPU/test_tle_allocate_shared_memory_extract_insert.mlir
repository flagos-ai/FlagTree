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

// RUN: triton-opt %s -split-input-file -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=120 ptx-version=88})' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @extract_tile_smem
  // CHECK: tle.extract_tile
  // CHECK-SAME: allocation.offset =
  tt.func @extract_tile_smem(%src: tensor<32x32xf32, #blocked>, %idx: i32) {
    %tile = tle.extract_tile %src[%idx] {tile_shape = array<i64: 16, 16>} : tensor<32x32xf32, #blocked>, i32 -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @insert_tile_smem
  // CHECK: tle.insert_tile
  // CHECK-SAME: allocation.offset =
  tt.func @insert_tile_smem(%src: tensor<32x32xf32, #blocked>, %tile: tensor<16x16xf32, #blocked>, %idx: i32) {
    %result = tle.insert_tile %src[%idx] = %tile {tile_shape = array<i64: 16, 16>} : tensor<32x32xf32, #blocked>, i32, tensor<16x16xf32, #blocked> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}
