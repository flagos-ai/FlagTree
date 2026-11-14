// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %s
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#nvmma3 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8, rank = 3}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Both accesses address bytes [256,384) of the same existing allocation.
  // Ignoring the second alias's byte origin incorrectly makes it [0,128).
  // CHECK-LABEL: llvm.func {{.*}}@owned_alias_byte_offset_raw
  // CHECK: {{st.shared|nvvm.stmatrix}}
  // CHECK: nvvm.barrier0
  // CHECK: {{ld.shared|nvvm.ldmatrix}}
  // CHECK: llvm.return
  tt.func private @owned_alias_byte_offset_raw(%value: tensor<32xi32, #blocked>) -> tensor<32xi32, #blocked> {
    %arena = ttg.local_alloc : () -> !ttg.memdesc<1024xi8, #shared, #smem, mutable>
    %a = tle.memdesc_alias %arena {offset_bytes = 0 : i64} : !ttg.memdesc<1024xi8, #shared, #smem, mutable> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    %b = tle.memdesc_alias %arena {offset_bytes = 256 : i64} : !ttg.memdesc<1024xi8, #shared, #smem, mutable> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    %range = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %c64 = arith.constant dense<64> : tensor<32xi32, #blocked>
    %upper = arith.addi %range, %c64 : tensor<32xi32, #blocked>
    %pa = "tle.local_pointers"(%a, %upper) : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<32xi32, #blocked>) -> tensor<32x!tt.ptr<i32, 3>, #blocked>
    %pb = "tle.local_pointers"(%b, %range) : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<32xi32, #blocked>) -> tensor<32x!tt.ptr<i32, 3>, #blocked>
    tt.store %pa, %value : tensor<32x!tt.ptr<i32, 3>, #blocked>
    %result = tt.load %pb : tensor<32x!tt.ptr<i32, 3>, #blocked>
    tt.return %result : tensor<32xi32, #blocked>
  }

  // CHECK-LABEL: llvm.func {{.*}}@owned_disjoint_nvmma_copies
  // CHECK: cp.async
  // CHECK-NOT: nvvm.barrier0
  // CHECK: llvm.return
  tt.func private @owned_disjoint_nvmma_copies(%source: tensor<64x128x!tt.ptr<i8>, #blocked2>) {
    %arena = ttg.local_alloc : () -> !ttg.memdesc<16384xi8, #shared, #smem, mutable>
    %a = tle.memdesc_alias %arena {offset_bytes = 0 : i64} : !ttg.memdesc<16384xi8, #shared, #smem, mutable> -> !ttg.memdesc<64x128xi8, #nvmma, #smem, mutable>
    %b = tle.memdesc_alias %arena {offset_bytes = 8192 : i64} : !ttg.memdesc<16384xi8, #shared, #smem, mutable> -> !ttg.memdesc<64x128xi8, #nvmma, #smem, mutable>
    %ta = ttg.async_copy_global_to_local %source, %a {contiguity = 4 : i32} : tensor<64x128x!tt.ptr<i8>, #blocked2> -> <64x128xi8, #nvmma, #smem, mutable>
    %tb = ttg.async_copy_global_to_local %source, %b {contiguity = 4 : i32} : tensor<64x128x!tt.ptr<i8>, #blocked2> -> <64x128xi8, #nvmma, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: llvm.func {{.*}}@owned_overlapping_nvmma_copies
  // CHECK: cp.async
  // CHECK: nvvm.barrier0
  // CHECK: cp.async
  // CHECK: llvm.return
  tt.func private @owned_overlapping_nvmma_copies(%source: tensor<64x128x!tt.ptr<i8>, #blocked2>) {
    %arena = ttg.local_alloc : () -> !ttg.memdesc<16384xi8, #shared, #smem, mutable>
    %a = tle.memdesc_alias %arena {offset_bytes = 0 : i64} : !ttg.memdesc<16384xi8, #shared, #smem, mutable> -> !ttg.memdesc<64x128xi8, #nvmma, #smem, mutable>
    %b = tle.memdesc_alias %arena {offset_bytes = 0 : i64} : !ttg.memdesc<16384xi8, #shared, #smem, mutable> -> !ttg.memdesc<64x128xi8, #nvmma, #smem, mutable>
    %ta = ttg.async_copy_global_to_local %source, %a {contiguity = 4 : i32} : tensor<64x128x!tt.ptr<i8>, #blocked2> -> <64x128xi8, #nvmma, #smem, mutable>
    %tb = ttg.async_copy_global_to_local %source, %b {contiguity = 4 : i32} : tensor<64x128x!tt.ptr<i8>, #blocked2> -> <64x128xi8, #nvmma, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: llvm.func {{.*}}@owned_dynamic_disjoint_nvmma_slots
  // CHECK: cp.async
  // CHECK-NOT: nvvm.barrier0
  // CHECK: llvm.return
  tt.func private @owned_dynamic_disjoint_nvmma_slots(%source: tensor<64x128x!tt.ptr<i8>, #blocked2>, %slot: i32) {
    %arena = ttg.local_alloc : () -> !ttg.memdesc<32768xi8, #shared, #smem, mutable>
    %a_slots = tle.memdesc_alias %arena {offset_bytes = 0 : i64} : !ttg.memdesc<32768xi8, #shared, #smem, mutable> -> !ttg.memdesc<2x64x128xi8, #nvmma3, #smem, mutable>
    %b_slots = tle.memdesc_alias %arena {offset_bytes = 16384 : i64} : !ttg.memdesc<32768xi8, #shared, #smem, mutable> -> !ttg.memdesc<2x64x128xi8, #nvmma3, #smem, mutable>
    %a = ttg.memdesc_index %a_slots[%slot] : !ttg.memdesc<2x64x128xi8, #nvmma3, #smem, mutable> -> !ttg.memdesc<64x128xi8, #nvmma, #smem, mutable>
    %b = ttg.memdesc_index %b_slots[%slot] : !ttg.memdesc<2x64x128xi8, #nvmma3, #smem, mutable> -> !ttg.memdesc<64x128xi8, #nvmma, #smem, mutable>
    %ta = ttg.async_copy_global_to_local %source, %a {contiguity = 4 : i32} : tensor<64x128x!tt.ptr<i8>, #blocked2> -> <64x128xi8, #nvmma, #smem, mutable>
    %tb = ttg.async_copy_global_to_local %source, %b {contiguity = 4 : i32} : tensor<64x128x!tt.ptr<i8>, #blocked2> -> <64x128xi8, #nvmma, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: llvm.func {{.*}}@owned_dynamic_overlapping_nvmma_slots
  // CHECK: cp.async
  // CHECK: nvvm.barrier0
  // CHECK: cp.async
  // CHECK: llvm.return
  tt.func private @owned_dynamic_overlapping_nvmma_slots(%source: tensor<64x128x!tt.ptr<i8>, #blocked2>, %slot: i32) {
    %arena = ttg.local_alloc : () -> !ttg.memdesc<32768xi8, #shared, #smem, mutable>
    %a_slots = tle.memdesc_alias %arena {offset_bytes = 0 : i64} : !ttg.memdesc<32768xi8, #shared, #smem, mutable> -> !ttg.memdesc<2x64x128xi8, #nvmma3, #smem, mutable>
    %b_slots = tle.memdesc_alias %arena {offset_bytes = 0 : i64} : !ttg.memdesc<32768xi8, #shared, #smem, mutable> -> !ttg.memdesc<2x64x128xi8, #nvmma3, #smem, mutable>
    %a = ttg.memdesc_index %a_slots[%slot] : !ttg.memdesc<2x64x128xi8, #nvmma3, #smem, mutable> -> !ttg.memdesc<64x128xi8, #nvmma, #smem, mutable>
    %b = ttg.memdesc_index %b_slots[%slot] : !ttg.memdesc<2x64x128xi8, #nvmma3, #smem, mutable> -> !ttg.memdesc<64x128xi8, #nvmma, #smem, mutable>
    %ta = ttg.async_copy_global_to_local %source, %a {contiguity = 4 : i32} : tensor<64x128x!tt.ptr<i8>, #blocked2> -> <64x128xi8, #nvmma, #smem, mutable>
    %tb = ttg.async_copy_global_to_local %source, %b {contiguity = 4 : i32} : tensor<64x128x!tt.ptr<i8>, #blocked2> -> <64x128xi8, #nvmma, #smem, mutable>
    tt.return
  }
}
