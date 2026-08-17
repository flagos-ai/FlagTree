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

// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1201 --convert-builtin-func-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @extract_tile_smem
  // CHECK: rocdl.workitem.id.x
  // CHECK-COUNT-2: rocdl.barrier
  // CHECK-NOT: tle.extract_tile
  // CHECK-NOT: nvvm.barrier
  tt.func @extract_tile_smem(%src: tensor<32x32xf32, #blocked>, %idx: i32) {
    %tile = tle.extract_tile %src[%idx] {tile_shape = array<i64: 16, 16>} : tensor<32x32xf32, #blocked>, i32 -> tensor<16x16xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @insert_tile_smem
  // CHECK: rocdl.workitem.id.x
  // CHECK-COUNT-2: rocdl.barrier
  // CHECK-NOT: tle.insert_tile
  // CHECK-NOT: nvvm.barrier
  tt.func @insert_tile_smem(%src: tensor<32x32xf32, #blocked>, %tile: tensor<16x16xf32, #blocked>, %idx: i32) {
    %result = tle.insert_tile %src[%idx] = %tile {tile_shape = array<i64: 16, 16>} : tensor<32x32xf32, #blocked>, i32, tensor<16x16xf32, #blocked> -> tensor<32x32xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @exclusive_cumsum
  // CHECK-COUNT-6: rocdl.ds_bpermute
  // CHECK-COUNT-2: rocdl.barrier
  // CHECK-NOT: tle.exclusive_cumsum
  // CHECK-NOT: nvvm.shfl
  tt.func public @exclusive_cumsum(%arg0: tensor<128xi32, #blocked>, %out: !tt.ptr<i32>) {
    %exclusive, %total = "tle.exclusive_cumsum"(%arg0) {axis = 0 : i32, reverse = false} : (tensor<128xi32, #blocked>) -> (tensor<128xi32, #blocked>, i32)
    tt.store %out, %total : !tt.ptr<i32>
    tt.return
  }
}
