// RUN: triton-opt %s --test-tritonhcugpu-membar -verify-each | FileCheck %s --enable-var-scope

// A barrier at the loop head orders the previous iteration's scratch reads.
// Two-atomic cases additionally require the within-iteration WAR barrier.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @rmw_loop_1_atomic(
  tt.func @rmw_loop_1_atomic(%ptr: !tt.ptr<i32>, %n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    // CHECK: scf.for
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_rmw{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 4 : i64
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%sum = %zero) -> (i32) {
      %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
      %next = arith.addi %sum, %a : i32
      scf.yield %next : i32
    }
    tt.return %result : i32
  }

  // CHECK-LABEL: @rmw_loop_2_atomic(
  tt.func @rmw_loop_2_atomic(%ptr: !tt.ptr<i32>, %n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    // CHECK: scf.for
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_rmw{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 4 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_rmw{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 4 : i64
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%sum = %zero) -> (i32) {
      %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
      %b = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
      %pair = arith.addi %a, %b : i32
      %next = arith.addi %sum, %pair : i32
      scf.yield %next : i32
    }
    tt.return %result : i32
  }

  // CHECK-LABEL: @cas_loop_1_atomic(
  tt.func @cas_loop_1_atomic(%ptr: !tt.ptr<i32>, %n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    // CHECK: scf.for
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 4 : i64
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%sum = %zero) -> (i32) {
      %a = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero {test.check_allocation} : (!tt.ptr<i32>, i32, i32) -> i32
      %next = arith.addi %sum, %a : i32
      scf.yield %next : i32
    }
    tt.return %result : i32
  }

  // CHECK-LABEL: @cas_loop_2_atomic(
  tt.func @cas_loop_2_atomic(%ptr: !tt.ptr<i32>, %n: index) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %zero = arith.constant 0 : i32
    // CHECK: scf.for
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 4 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 4 : i64
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%sum = %zero) -> (i32) {
      %a = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero {test.check_allocation} : (!tt.ptr<i32>, i32, i32) -> i32
      %b = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero {test.check_allocation} : (!tt.ptr<i32>, i32, i32) -> i32
      %pair = arith.addi %a, %b : i32
      %next = arith.addi %sum, %pair : i32
      scf.yield %next : i32
    }
    tt.return %result : i32
  }
}
