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
