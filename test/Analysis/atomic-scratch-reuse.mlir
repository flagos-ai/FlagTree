// RUN: triton-opt %s --test-print-membar -verify-each | FileCheck %s
// The HCU triton-opt exposes this analysis as --test-tritonhcugpu-membar.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @add_scratch_reuse
  tt.func @add_scratch_reuse(%ptr: !tt.ptr<i32>) -> i32 {
    %one = arith.constant 1 : i32
    // CHECK: tt.atomic_rmw add
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_rmw add
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32
    %b = tt.atomic_rmw add, relaxed, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32
    %sum = arith.addi %a, %b : i32
    tt.return %sum : i32
  }

  // CHECK-LABEL: @cas_scratch_reuse
  tt.func @cas_scratch_reuse(%ptr: !tt.ptr<i32>) -> i32 {
    %zero = arith.constant 0 : i32
    // CHECK: tt.atomic_cas
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas
    %a = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero : (!tt.ptr<i32>, i32, i32) -> i32
    %b = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero : (!tt.ptr<i32>, i32, i32) -> i32
    %sum = arith.addi %a, %b : i32
    tt.return %sum : i32
  }

  tt.func private @returned_add(%ptr: !tt.ptr<i32>) -> i32 {
    %one = arith.constant 1 : i32
    %value = tt.atomic_rmw add, relaxed, gpu, %ptr, %one : (!tt.ptr<i32>, i32) -> i32
    tt.return %value : i32
  }

  // Both summaries refer to a callee atomic, absent from the caller's Allocation.
  // CHECK-LABEL: @call_scratch_reuse
  // CHECK: tt.call @returned_add
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: %{{.*}} = tt.call @returned_add
  tt.func @call_scratch_reuse(%ptr: !tt.ptr<i32>) -> i32 {
    %a = tt.call @returned_add(%ptr) : (!tt.ptr<i32>) -> i32
    %b = tt.call @returned_add(%ptr) : (!tt.ptr<i32>) -> i32
    %sum = arith.addi %a, %b : i32
    tt.return %sum : i32
  }

  // Unused results need no scratch broadcast or additional barrier.
  // CHECK-LABEL: @no_scratch
  // CHECK-NOT: ttg.local_barrier
  // CHECK: tt.atomic_rmw add
  // CHECK-NOT: ttg.local_barrier
  // CHECK: tt.atomic_cas
  // CHECK-NOT: ttg.local_barrier
  // CHECK: tt.return
  tt.func @no_scratch(%ptr: !tt.ptr<i32>) {
    %zero = arith.constant 0 : i32
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero : (!tt.ptr<i32>, i32) -> i32
    %b = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero : (!tt.ptr<i32>, i32, i32) -> i32
    tt.return
  }
}
