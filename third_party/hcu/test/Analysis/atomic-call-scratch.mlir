// RUN: triton-opt %s --test-tritonhcugpu-membar -verify-each | FileCheck %s
// Keep a 4096-byte allocation live across the accesses. Check the allocator's
// actual offsets and sizes, not only the presence of a barrier.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @leaf(
  tt.func private @leaf(%ptr: !tt.ptr<i32>) -> i32 {
    %one = arith.constant 1 : i32
    // CHECK: tt.atomic_rmw add{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 4 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %one {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
    tt.return %a : i32
  }

  // CHECK-LABEL: @trailing(
  tt.func private @trailing(%ptr: !tt.ptr<i32>) -> i32 {
    %one = arith.constant 1 : i32
    // CHECK: tt.atomic_rmw add{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 4 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %one {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
    ttg.local_barrier
    tt.return %a : i32
  }

  // CHECK-LABEL: @leading(
  tt.func private @leading(%ptr: !tt.ptr<i32>) -> i32 {
    %one = arith.constant 1 : i32
    ttg.local_barrier
    // CHECK: tt.atomic_rmw add{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 4 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %one {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
    tt.return %a : i32
  }

  // CHECK-LABEL: @nested(
  tt.func private @nested(%ptr: !tt.ptr<i32>) -> i32 {
    // CHECK: tt.call @leaf{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 4 : i64
    %a = tt.call @leaf(%ptr) {test.check_allocation} : (!tt.ptr<i32>) -> i32
    tt.return %a : i32
  }

  // CHECK-LABEL: @caller_to_callee(
  tt.func @caller_to_callee(%ptr: !tt.ptr<i32>) -> (i32, tensor<1024xi32, #blocked>) {
    %one = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable>
    // CHECK: tt.atomic_rmw add{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.call @leaf{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %one {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
    %b = tt.call @leaf(%ptr) {test.check_allocation} : (!tt.ptr<i32>) -> i32
    %sum = arith.addi %a, %b : i32
    %keep = ttg.local_load %buf : !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable> -> tensor<1024xi32, #blocked>
    tt.return %sum, %keep : i32, tensor<1024xi32, #blocked>
  }

  // CHECK-LABEL: @callee_to_caller(
  tt.func @callee_to_caller(%ptr: !tt.ptr<i32>) -> (i32, tensor<1024xi32, #blocked>) {
    %one = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable>
    // CHECK: tt.call @leaf{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_rmw add{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    %b = tt.call @leaf(%ptr) {test.check_allocation} : (!tt.ptr<i32>) -> i32
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %one {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
    %sum = arith.addi %a, %b : i32
    %keep = ttg.local_load %buf : !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable> -> tensor<1024xi32, #blocked>
    tt.return %sum, %keep : i32, tensor<1024xi32, #blocked>
  }

  // CHECK-LABEL: @nested_to_caller(
  tt.func @nested_to_caller(%ptr: !tt.ptr<i32>) -> (i32, tensor<1024xi32, #blocked>) {
    %one = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable>
    // CHECK: tt.call @nested{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_rmw add{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    %b = tt.call @nested(%ptr) {test.check_allocation} : (!tt.ptr<i32>) -> i32
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %one {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
    %sum = arith.addi %a, %b : i32
    %keep = ttg.local_load %buf : !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable> -> tensor<1024xi32, #blocked>
    tt.return %sum, %keep : i32, tensor<1024xi32, #blocked>
  }

  // CHECK-LABEL: @caller_to_nested(
  tt.func @caller_to_nested(%ptr: !tt.ptr<i32>) -> (i32, tensor<1024xi32, #blocked>) {
    %one = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable>
    // CHECK: tt.atomic_rmw add{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.call @nested{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %one {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
    %b = tt.call @nested(%ptr) {test.check_allocation} : (!tt.ptr<i32>) -> i32
    %sum = arith.addi %a, %b : i32
    %keep = ttg.local_load %buf : !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable> -> tensor<1024xi32, #blocked>
    tt.return %sum, %keep : i32, tensor<1024xi32, #blocked>
  }

  // CHECK-LABEL: @callee_trailing_barrier(
  tt.func @callee_trailing_barrier(%ptr: !tt.ptr<i32>) -> (i32, tensor<1024xi32, #blocked>) {
    %one = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable>
    // CHECK: tt.atomic_rmw add{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.call @trailing{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %one {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
    %b = tt.call @trailing(%ptr) {test.check_allocation} : (!tt.ptr<i32>) -> i32
    %sum = arith.addi %a, %b : i32
    %keep = ttg.local_load %buf : !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable> -> tensor<1024xi32, #blocked>
    tt.return %sum, %keep : i32, tensor<1024xi32, #blocked>
  }

  // CHECK-LABEL: @callee_leading_barrier(
  tt.func @callee_leading_barrier(%ptr: !tt.ptr<i32>) -> (i32, tensor<1024xi32, #blocked>) {
    %one = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable>
    // CHECK: tt.atomic_rmw add{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    // CHECK-NEXT: %{{.*}} = tt.call @leading{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 4 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %one {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
    %b = tt.call @leading(%ptr) {test.check_allocation} : (!tt.ptr<i32>) -> i32
    %sum = arith.addi %a, %b : i32
    %keep = ttg.local_load %buf : !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable> -> tensor<1024xi32, #blocked>
    tt.return %sum, %keep : i32, tensor<1024xi32, #blocked>
  }

  // Two nonzero call offsets must compose: 4096 + 1024 = 5120.
  // CHECK-LABEL: @nested_offset(
  tt.func private @nested_offset(%ptr: !tt.ptr<i32>) -> (i32, tensor<256xi32, #blocked>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #ttg.shared_memory, mutable>
    // CHECK: tt.call @leaf{{.*}}test.buffer_offset = 1024 : i64, test.buffer_size = 4 : i64
    %a = tt.call @leaf(%ptr) {test.check_allocation} : (!tt.ptr<i32>) -> i32
    %keep = ttg.local_load %buf : !ttg.memdesc<256xi32, #shared, #ttg.shared_memory, mutable> -> tensor<256xi32, #blocked>
    tt.return %a, %keep : i32, tensor<256xi32, #blocked>
  }

  // CHECK-LABEL: @nested_nonzero_offsets(
  tt.func @nested_nonzero_offsets(%ptr: !tt.ptr<i32>) -> (i32, tensor<1024xi32, #blocked>, tensor<256xi32, #blocked>) {
    %one = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable>
    // The temporary occupies the first 1024 bytes of the future call frame.
    %tmp = ttg.local_alloc : () -> !ttg.memdesc<256xi32, #shared, #ttg.shared_memory, mutable>
    // CHECK: tt.atomic_rmw add{{.*}}test.buffer_offset = 5120 : i64, test.buffer_size = 4 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %one {test.check_allocation} : (!tt.ptr<i32>, i32) -> i32
    // CHECK-NEXT: %{{.*}} = ttg.local_load
    %saved = ttg.local_load %tmp : !ttg.memdesc<256xi32, #shared, #ttg.shared_memory, mutable> -> tensor<256xi32, #blocked>
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.call @nested_offset{{.*}}test.buffer_offset = 4096 : i64, test.buffer_size = 1028 : i64
    %b, %inner = tt.call @nested_offset(%ptr) {test.check_allocation} : (!tt.ptr<i32>) -> (i32, tensor<256xi32, #blocked>)
    %sum = arith.addi %a, %b : i32
    %keep = ttg.local_load %buf : !ttg.memdesc<1024xi32, #shared, #ttg.shared_memory, mutable> -> tensor<1024xi32, #blocked>
    tt.return %sum, %keep, %saved : i32, tensor<1024xi32, #blocked>, tensor<256xi32, #blocked>
  }
}
