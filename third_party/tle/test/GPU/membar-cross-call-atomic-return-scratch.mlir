// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s --test-print-membar | FileCheck %s
// RUN: triton-opt %s --test-tritonamdgpu-membar | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @returned_rmw(%global: !tt.ptr<i64>) -> i64 {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %value = tt.atomic_rmw add, relaxed, sys, %global, %zero, %true : (!tt.ptr<i64>, i64, i1) -> i64
    tt.return %value : i64
  }

  tt.func private @returned_cas(%global: !tt.ptr<i64>) -> i64 {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %value = tt.atomic_cas acq_rel, sys, %global, %zero, %zero : (!tt.ptr<i64>, i64, i64) -> i64
    tt.return %value : i64
  }

  // A summary keeps callee operations. Their broadcast scratch belongs to
  // the callee Allocation even after its interval is translated to the caller.
  // CHECK-LABEL: @caller_rmw_return_then_shared_rmw
  // CHECK: tt.call @returned_rmw
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: %{{.*}} = tt.atomic_rmw
  tt.func @caller_rmw_return_then_shared_rmw(%shared: !tt.ptr<i64, 3>, %global: !tt.ptr<i64>) -> i64 {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %value = tt.call @returned_rmw(%global) : (!tt.ptr<i64>) -> i64
    %unused = tt.atomic_rmw add, relaxed, cta, %shared, %zero, %true : (!tt.ptr<i64, 3>, i64, i1) -> i64
    tt.return %value : i64
  }

  // CHECK-LABEL: @caller_rmw_return_then_shared_cas
  // CHECK: tt.call @returned_rmw
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: %{{.*}} = tt.atomic_cas
  tt.func @caller_rmw_return_then_shared_cas(%shared: !tt.ptr<i64, 3>, %global: !tt.ptr<i64>) -> i64 {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %value = tt.call @returned_rmw(%global) : (!tt.ptr<i64>) -> i64
    %unused = tt.atomic_cas acq_rel, cta, %shared, %zero, %zero : (!tt.ptr<i64, 3>, i64, i64) -> i64
    tt.return %value : i64
  }

  // CHECK-LABEL: @caller_cas_return_then_shared_rmw
  // CHECK: tt.call @returned_cas
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: %{{.*}} = tt.atomic_rmw
  tt.func @caller_cas_return_then_shared_rmw(%shared: !tt.ptr<i64, 3>, %global: !tt.ptr<i64>) -> i64 {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %value = tt.call @returned_cas(%global) : (!tt.ptr<i64>) -> i64
    %unused = tt.atomic_rmw add, relaxed, cta, %shared, %zero, %true : (!tt.ptr<i64, 3>, i64, i1) -> i64
    tt.return %value : i64
  }

  // CHECK-LABEL: @caller_cas_return_then_shared_cas
  // CHECK: tt.call @returned_cas
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: %{{.*}} = tt.atomic_cas
  tt.func @caller_cas_return_then_shared_cas(%shared: !tt.ptr<i64, 3>, %global: !tt.ptr<i64>) -> i64 {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %value = tt.call @returned_cas(%global) : (!tt.ptr<i64>) -> i64
    %unused = tt.atomic_cas acq_rel, cta, %shared, %zero, %zero : (!tt.ptr<i64, 3>, i64, i64) -> i64
    tt.return %value : i64
  }

  tt.func private @forward_return(%global: !tt.ptr<i64>) -> i64 {
    %value = tt.call @returned_rmw(%global) : (!tt.ptr<i64>) -> i64
    tt.return %value : i64
  }

  // CHECK-LABEL: @nested_return_then_shared_atomic
  // CHECK: tt.call @forward_return
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: %{{.*}} = tt.atomic_rmw
  tt.func @nested_return_then_shared_atomic(%shared: !tt.ptr<i64, 3>, %global: !tt.ptr<i64>) -> i64 {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %value = tt.call @forward_return(%global) : (!tt.ptr<i64>) -> i64
    %unused = tt.atomic_rmw add, relaxed, cta, %shared, %zero, %true : (!tt.ptr<i64, 3>, i64, i1) -> i64
    tt.return %value : i64
  }

  tt.func private @no_return_shared_atomic(%shared: !tt.ptr<i64, 3>) {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %unused = tt.atomic_rmw add, relaxed, cta, %shared, %zero, %true : (!tt.ptr<i64, 3>, i64, i1) -> i64
    tt.return
  }

  // Atomicity is enough when both operations have no return broadcast.
  // CHECK-LABEL: @no_scratch_atomic_chain
  // CHECK: tt.call @no_return_shared_atomic
  // CHECK-NOT: ttg.local_barrier
  // CHECK: tt.atomic_rmw
  // CHECK-NOT: ttg.local_barrier
  // CHECK: tt.return
  tt.func @no_scratch_atomic_chain(%shared: !tt.ptr<i64, 3>) {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    tt.call @no_return_shared_atomic(%shared) : (!tt.ptr<i64, 3>) -> ()
    %unused = tt.atomic_rmw add, relaxed, cta, %shared, %zero, %true : (!tt.ptr<i64, 3>, i64, i1) -> i64
    tt.return
  }
}
