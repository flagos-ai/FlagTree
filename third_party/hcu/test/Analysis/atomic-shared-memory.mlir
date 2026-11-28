// RUN: triton-opt %s --test-tritonhcugpu-membar -verify-each | FileCheck %s --enable-var-scope

// Both atomic addresses alias the same allocated shared-memory element.
// Verify target intervals independently of return-broadcast scratch.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @shared_atomic_returns(
  tt.func @shared_atomic_returns() -> (i32, i32) {
    %zero = arith.constant 0 : i32
    %smem = ttg.local_alloc : () -> !ttg.memdesc<4xi32, #shared, #ttg.shared_memory, mutable>
    %ptr = "tle.local_pointers"(%smem, %zero) : (!ttg.memdesc<4xi32, #shared, #ttg.shared_memory, mutable>, i32) -> !tt.ptr<i32, 3>
    // CHECK: tt.atomic_rmw{{.*}}test.buffer_offset = 128 : i64, test.buffer_size = 4 : i64{{.*}}test.has_scratch = true, test.shared_offset = 0 : i64, test.shared_size = 16 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas{{.*}}test.buffer_offset = 128 : i64, test.buffer_size = 4 : i64{{.*}}test.has_scratch = true, test.shared_offset = 0 : i64, test.shared_size = 16 : i64
    // CHECK-NOT: ttg.local_barrier
    // CHECK: tt.return
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero {test.check_allocation, test.check_shared} : (!tt.ptr<i32, 3>, i32) -> i32
    %b = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero {test.check_allocation, test.check_shared} : (!tt.ptr<i32, 3>, i32, i32) -> i32
    tt.return %a, %b : i32, i32
  }

  // CHECK-LABEL: @shared_atomic_no_returns(
  tt.func @shared_atomic_no_returns() {
    %zero = arith.constant 0 : i32
    %smem = ttg.local_alloc : () -> !ttg.memdesc<4xi32, #shared, #ttg.shared_memory, mutable>
    %ptr = "tle.local_pointers"(%smem, %zero) : (!ttg.memdesc<4xi32, #shared, #ttg.shared_memory, mutable>, i32) -> !tt.ptr<i32, 3>
    // CHECK: tt.atomic_rmw{{.*}}test.has_scratch = false, test.shared_offset = 0 : i64, test.shared_size = 16 : i64
    // CHECK-NOT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas{{.*}}test.has_scratch = false, test.shared_offset = 0 : i64, test.shared_size = 16 : i64
    // CHECK-NOT: ttg.local_barrier
    // CHECK: tt.return
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero {test.check_shared} : (!tt.ptr<i32, 3>, i32) -> i32
    %b = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero {test.check_shared} : (!tt.ptr<i32, 3>, i32, i32) -> i32
    tt.return
  }

  // CHECK-LABEL: @shared_only_rmw_return(
  tt.func @shared_only_rmw_return() -> i32 {
    %zero = arith.constant 0 : i32
    %smem = ttg.local_alloc : () -> !ttg.memdesc<4xi32, #shared, #ttg.shared_memory, mutable>
    %ptr = "tle.local_pointers"(%smem, %zero) : (!ttg.memdesc<4xi32, #shared, #ttg.shared_memory, mutable>, i32) -> !tt.ptr<i32, 3>
    // CHECK: tt.atomic_rmw{{.*}}test.has_scratch = true, test.shared_offset = 0 : i64, test.shared_size = 16 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas{{.*}}test.has_scratch = false, test.shared_offset = 0 : i64, test.shared_size = 16 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero {test.check_shared} : (!tt.ptr<i32, 3>, i32) -> i32
    %b = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero {test.check_shared} : (!tt.ptr<i32, 3>, i32, i32) -> i32
    tt.return %a : i32
  }

  // CHECK-LABEL: @shared_only_cas_return(
  tt.func @shared_only_cas_return() -> i32 {
    %zero = arith.constant 0 : i32
    %smem = ttg.local_alloc : () -> !ttg.memdesc<4xi32, #shared, #ttg.shared_memory, mutable>
    %ptr = "tle.local_pointers"(%smem, %zero) : (!ttg.memdesc<4xi32, #shared, #ttg.shared_memory, mutable>, i32) -> !tt.ptr<i32, 3>
    // CHECK: tt.atomic_rmw{{.*}}test.has_scratch = false, test.shared_offset = 0 : i64, test.shared_size = 16 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas{{.*}}test.has_scratch = true, test.shared_offset = 0 : i64, test.shared_size = 16 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero {test.check_shared} : (!tt.ptr<i32, 3>, i32) -> i32
    %b = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero {test.check_shared} : (!tt.ptr<i32, 3>, i32, i32) -> i32
    tt.return %b : i32
  }
}
