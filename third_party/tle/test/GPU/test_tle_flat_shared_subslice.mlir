// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=120 ptx-version=88' --convert-triton-gpu-to-llvm='compute-capability=120 ptx-version=88' -reconcile-unrealized-casts | FileCheck %s

#flat = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:120", tle.static_shared = 1 : i32} {
  // The slice starts at element 96; the alias adds one element. The byte
  // origin is 384 + 4, with the original 768-byte backing still reserved.
  tt.func public @flat_subslice_alias() {
    %zero = arith.constant 0 : i32
    %s = ttg.local_alloc {alignment = 16 : i32} : () -> !ttg.memdesc<192xf32, #flat, #smem, mutable>
    %v = ttg.memdesc_subslice %s[96] : !ttg.memdesc<192xf32, #flat, #smem, mutable> -> !ttg.memdesc<96xf32, #flat, #smem, mutable, 192>
    %a = tle.memdesc_alias %v {offset_bytes = 4 : i64} : !ttg.memdesc<96xf32, #flat, #smem, mutable, 192> -> !ttg.memdesc<32xf32, #flat, #smem, mutable>
    %p = "tle.local_pointers"(%a, %zero) : (!ttg.memdesc<32xf32, #flat, #smem, mutable>, i32) -> !tt.ptr<f32, 3>
    %value = tt.load %p : !tt.ptr<f32, 3>
    tt.return
  }
}
// CHECK: ttg.static_shared = 768
// CHECK: llvm.mlir.global internal @global_smem
// CHECK-LABEL: llvm.func @flat_subslice_alias
// CHECK: llvm.mlir.constant(96 : i32)
// CHECK: llvm.mlir.constant(4 : i32)
// CHECK: ld.shared.b32
