// MIT License

// Copyright (c) 2025 The FlagOS Contributors

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// flagtree tle

// RUN: triton-opt %s -split-input-file -triton-tle-lower-tma-copy | FileCheck %s

// Test TLE TMA copy operations lowering to NVIDIA TMA operations
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [2, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @elementwise_tma_add_kernel(%a_desc: !tt.tensordesc<tensor<32x64xf32, #shared>>, %a_desc_0: i32, %a_desc_1: i32, %a_desc_2: i64, %a_desc_3: i64, %b_desc: !tt.tensordesc<tensor<32x64xf32, #shared>>, %b_desc_4: i32, %b_desc_5: i32, %b_desc_6: i64, %b_desc_7: i64, %c_desc: !tt.tensordesc<tensor<32x64xf32, #shared>>, %c_desc_8: i32, %c_desc_9: i32, %c_desc_10: i64, %c_desc_11: i64, %xnumel: i32 {tt.divisibility = 16 : i32}, %ynumel: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %true = arith.constant true
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %c32_i32 = arith.constant 32 : i32
    %pid = tt.get_program_id x : i32

    // Test TLE local_alloc operations
    // CHECK: ttg.local_alloc
    // CHECK-SAME: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
    %a_smem = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
    %b_smem = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
    %c_smem = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>

    %0 = arith.muli %pid, %c32_i32 : i32
    scf.for %yoff = %c0_i32 to %ynumel step %c64_i32  : i32 {
      // Test TLE tma_copy operations (global to shared) - should be lowered to NVIDIA TMA operations
      // CHECK: [[BARRIER1:%.+]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1
      // CHECK: ttng.init_barrier [[BARRIER1]], 1
      // CHECK: ttng.barrier_expect [[BARRIER1]], 8192, %true
      // CHECK: ttng.async_tma_copy_global_to_local {{.+}}[{{.+}}] {{.+}} [[BARRIER1]], %true
      // CHECK: ttng.wait_barrier [[BARRIER1]], %c0_i32
      // CHECK: ttng.inval_barrier [[BARRIER1]]
      ttg.tma_copy %a_desc, %a_smem, [%0, %yoff] : !tt.tensordesc<tensor<32x64xf32, #shared>>, !ttg.memdesc<32x64xf32, #shared, #smem, mutable>

      // CHECK: [[BARRIER2:%.+]] = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #shared1
      // CHECK: ttng.init_barrier [[BARRIER2]], 1
      // CHECK: ttng.barrier_expect [[BARRIER2]], 8192, %true
      // CHECK: ttng.async_tma_copy_global_to_local {{.+}}[{{.+}}] {{.+}} [[BARRIER2]], %true
      // CHECK: ttng.wait_barrier [[BARRIER2]], %c0_i32
      // CHECK: ttng.inval_barrier [[BARRIER2]]
      ttg.tma_copy %b_desc, %b_smem, [%0, %yoff] : !tt.tensordesc<tensor<32x64xf32, #shared>>, !ttg.memdesc<32x64xf32, #shared, #smem, mutable>

      // Test TLE local_load operations
      // CHECK: ttg.local_load
      // CHECK-SAME: !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
      %c_val = ttg.local_load %a_smem : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> tensor<32x64xf32, #blocked>
      %c_val_12 = ttg.local_load %b_smem : !ttg.memdesc<32x64xf32, #shared, #smem, mutable> -> tensor<32x64xf32, #blocked>

      %c_val_13 = arith.addf %c_val, %c_val_12 : tensor<32x64xf32, #blocked>

      // Test TLE local_store operation
      // CHECK: ttg.local_store
      // CHECK-SAME: tensor<32x64xf32, #blocked> -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>
      ttg.local_store %c_val_13, %c_smem : tensor<32x64xf32, #blocked> -> !ttg.memdesc<32x64xf32, #shared, #smem, mutable>

      // Test TLE tma_copy operation (shared to global) - should be lowered to NVIDIA async TMA store
      // CHECK: ttng.fence_async_shared
      // CHECK: ttng.async_tma_copy_local_to_global {{.+}}[{{.+}}] {{.+}}
      // CHECK: ttng.async_tma_store_wait
      ttg.tma_copy %c_smem, %c_desc, [%0, %yoff] : !ttg.memdesc<32x64xf32, #shared, #smem, mutable>, !tt.tensordesc<tensor<32x64xf32, #shared>>
    }
    tt.return
  }
}
