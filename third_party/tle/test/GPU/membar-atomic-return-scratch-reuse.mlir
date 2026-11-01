// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s --test-print-membar -verify-each | FileCheck %s
//
// Atomicity of the addressed global memory does not order the shared-memory
// broadcasts of returned values. Their scratch slots can overlap, so readers
// of the previous return must finish before the next return overwrites them.
// Minimal GPU fail-before: r1830 scalar-to-four-peer atomic returns, W4/W8,
// 16 CTA and 16 Graph replays: racecheck exit86, 286 errors/4100 warnings.
// No model weights, communication SDK, or application fences are involved.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [8], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @rmw_to_rmw
  // CHECK: tt.atomic_rmw
  // CHECK: ttg.local_barrier
  // CHECK-NEXT: %{{.*}} = tt.atomic_rmw
  tt.func public @rmw_to_rmw(%ready: !tt.ptr<i64>, %peers: tensor<4x!tt.ptr<i64>, #blocked>) -> (i64, tensor<4xi64, #blocked>) {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %zeros = arith.constant dense<0> : tensor<4xi64, #blocked>
    %all = arith.constant dense<true> : tensor<4xi1, #blocked>
    %first = tt.atomic_rmw add, relaxed, sys, %ready, %zero, %true : (!tt.ptr<i64>, i64, i1) -> i64
    %second = tt.atomic_rmw add, acquire, sys, %peers, %zeros, %all : (tensor<4x!tt.ptr<i64>, #blocked>, tensor<4xi64, #blocked>, tensor<4xi1, #blocked>) -> tensor<4xi64, #blocked>
    tt.return %first, %second : i64, tensor<4xi64, #blocked>
  }

  // CHECK-LABEL: @cas_to_rmw
  // CHECK: tt.atomic_cas
  // CHECK: ttg.local_barrier
  // CHECK-NEXT: %{{.*}} = tt.atomic_rmw
  tt.func public @cas_to_rmw(%ready: !tt.ptr<i64>, %peers: tensor<4x!tt.ptr<i64>, #blocked>) -> (i64, tensor<4xi64, #blocked>) {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %zeros = arith.constant dense<0> : tensor<4xi64, #blocked>
    %all = arith.constant dense<true> : tensor<4xi1, #blocked>
    %first = tt.atomic_cas acq_rel, sys, %ready, %zero, %zero : (!tt.ptr<i64>, i64, i64) -> i64
    %second = tt.atomic_rmw add, acquire, sys, %peers, %zeros, %all : (tensor<4x!tt.ptr<i64>, #blocked>, tensor<4xi64, #blocked>, tensor<4xi1, #blocked>) -> tensor<4xi64, #blocked>
    tt.return %first, %second : i64, tensor<4xi64, #blocked>
  }

  // CHECK-LABEL: @rmw_to_cas
  // CHECK: tt.atomic_rmw
  // CHECK: ttg.local_barrier
  // CHECK-NEXT: %{{.*}} = tt.atomic_cas
  tt.func public @rmw_to_cas(%ready: !tt.ptr<i64>, %peers: tensor<4x!tt.ptr<i64>, #blocked>) -> (i64, tensor<4xi64, #blocked>) {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %zeros = arith.constant dense<0> : tensor<4xi64, #blocked>
    %all = arith.constant dense<true> : tensor<4xi1, #blocked>
    %first = tt.atomic_rmw add, relaxed, sys, %ready, %zero, %true : (!tt.ptr<i64>, i64, i1) -> i64
    %second = tt.atomic_cas acq_rel, sys, %peers, %zeros, %zeros : (tensor<4x!tt.ptr<i64>, #blocked>, tensor<4xi64, #blocked>, tensor<4xi64, #blocked>) -> tensor<4xi64, #blocked>
    tt.return %first, %second : i64, tensor<4xi64, #blocked>
  }

  // CHECK-LABEL: @cas_to_cas
  // CHECK: tt.atomic_cas
  // CHECK: ttg.local_barrier
  // CHECK-NEXT: %{{.*}} = tt.atomic_cas
  tt.func public @cas_to_cas(%ready: !tt.ptr<i64>, %peers: tensor<4x!tt.ptr<i64>, #blocked>) -> (i64, tensor<4xi64, #blocked>) {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %zeros = arith.constant dense<0> : tensor<4xi64, #blocked>
    %all = arith.constant dense<true> : tensor<4xi1, #blocked>
    %first = tt.atomic_cas acq_rel, sys, %ready, %zero, %zero : (!tt.ptr<i64>, i64, i64) -> i64
    %second = tt.atomic_cas acq_rel, sys, %peers, %zeros, %zeros : (tensor<4x!tt.ptr<i64>, #blocked>, tensor<4xi64, #blocked>, tensor<4xi64, #blocked>) -> tensor<4xi64, #blocked>
    tt.return %first, %second : i64, tensor<4xi64, #blocked>
  }

  // The no-return global atomic chain has no shared broadcast scratch.
  // CHECK-LABEL: @unused_global_atomic_returns
  // CHECK: tt.atomic_rmw
  // CHECK-NOT: ttg.local_barrier
  // CHECK: tt.atomic_cas
  // CHECK-NOT: ttg.local_barrier
  // CHECK: tt.return
  tt.func public @unused_global_atomic_returns(%ready: !tt.ptr<i64>) {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %a = tt.atomic_rmw add, relaxed, sys, %ready, %zero, %true : (!tt.ptr<i64>, i64, i1) -> i64
    %b = tt.atomic_cas acq_rel, sys, %ready, %zero, %zero : (!tt.ptr<i64>, i64, i64) -> i64
    tt.return
  }
}
