// RUN: triton-opt %s --test-tritonhcugpu-membar -verify-each | FileCheck %s --enable-var-scope

// Check 8-byte scalar scratch against both scalar and replicated tensor returns.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [64], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: @rmw_to_rmw_i64(
  tt.func @rmw_to_rmw_i64(%ptr: !tt.ptr<i64>, %next: !tt.ptr<i64>) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    // CHECK: tt.atomic_rmw{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_rmw{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero {test.check_allocation} : (!tt.ptr<i64>, i64) -> i64
    %b = tt.atomic_rmw add, relaxed, gpu, %next, %zero {test.check_allocation} : (!tt.ptr<i64>, i64) -> i64
    tt.return %a, %b : i64, i64
  }

  // CHECK-LABEL: @rmw_to_cas_i64(
  tt.func @rmw_to_cas_i64(%ptr: !tt.ptr<i64>, %next: !tt.ptr<i64>) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    // CHECK: tt.atomic_rmw{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero {test.check_allocation} : (!tt.ptr<i64>, i64) -> i64
    %b = tt.atomic_cas acq_rel, gpu, %next, %zero, %zero {test.check_allocation} : (!tt.ptr<i64>, i64, i64) -> i64
    tt.return %a, %b : i64, i64
  }

  // CHECK-LABEL: @cas_to_rmw_i64(
  tt.func @cas_to_rmw_i64(%ptr: !tt.ptr<i64>, %next: !tt.ptr<i64>) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    // CHECK: tt.atomic_cas{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_rmw{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    %a = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero {test.check_allocation} : (!tt.ptr<i64>, i64, i64) -> i64
    %b = tt.atomic_rmw add, relaxed, gpu, %next, %zero {test.check_allocation} : (!tt.ptr<i64>, i64) -> i64
    tt.return %a, %b : i64, i64
  }

  // CHECK-LABEL: @cas_to_cas_i64(
  tt.func @cas_to_cas_i64(%ptr: !tt.ptr<i64>, %next: !tt.ptr<i64>) -> (i64, i64) {
    %zero = arith.constant 0 : i64
    // CHECK: tt.atomic_cas{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    %a = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero {test.check_allocation} : (!tt.ptr<i64>, i64, i64) -> i64
    %b = tt.atomic_cas acq_rel, gpu, %next, %zero, %zero {test.check_allocation} : (!tt.ptr<i64>, i64, i64) -> i64
    tt.return %a, %b : i64, i64
  }

  // CHECK-LABEL: @rmw_to_rmw_i64_tensor(
  tt.func @rmw_to_rmw_i64_tensor(%ptr: !tt.ptr<i64>, %next: tensor<4x!tt.ptr<i64>, #blocked>) -> (i64, tensor<4xi64, #blocked>) {
    %zero = arith.constant 0 : i64
    %zeros = arith.constant dense<0> : tensor<4xi64, #blocked>
    // CHECK: tt.atomic_rmw{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_rmw{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 32 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero {test.check_allocation} : (!tt.ptr<i64>, i64) -> i64
    %b = tt.atomic_rmw add, relaxed, gpu, %next, %zeros {test.check_allocation} : (tensor<4x!tt.ptr<i64>, #blocked>, tensor<4xi64, #blocked>) -> tensor<4xi64, #blocked>
    tt.return %a, %b : i64, tensor<4xi64, #blocked>
  }

  // CHECK-LABEL: @rmw_to_cas_i64_tensor(
  tt.func @rmw_to_cas_i64_tensor(%ptr: !tt.ptr<i64>, %next: tensor<4x!tt.ptr<i64>, #blocked>) -> (i64, tensor<4xi64, #blocked>) {
    %zero = arith.constant 0 : i64
    %zeros = arith.constant dense<0> : tensor<4xi64, #blocked>
    // CHECK: tt.atomic_rmw{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 32 : i64
    %a = tt.atomic_rmw add, relaxed, gpu, %ptr, %zero {test.check_allocation} : (!tt.ptr<i64>, i64) -> i64
    %b = tt.atomic_cas acq_rel, gpu, %next, %zeros, %zeros {test.check_allocation} : (tensor<4x!tt.ptr<i64>, #blocked>, tensor<4xi64, #blocked>, tensor<4xi64, #blocked>) -> tensor<4xi64, #blocked>
    tt.return %a, %b : i64, tensor<4xi64, #blocked>
  }

  // CHECK-LABEL: @cas_to_rmw_i64_tensor(
  tt.func @cas_to_rmw_i64_tensor(%ptr: !tt.ptr<i64>, %next: tensor<4x!tt.ptr<i64>, #blocked>) -> (i64, tensor<4xi64, #blocked>) {
    %zero = arith.constant 0 : i64
    %zeros = arith.constant dense<0> : tensor<4xi64, #blocked>
    // CHECK: tt.atomic_cas{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_rmw{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 32 : i64
    %a = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero {test.check_allocation} : (!tt.ptr<i64>, i64, i64) -> i64
    %b = tt.atomic_rmw add, relaxed, gpu, %next, %zeros {test.check_allocation} : (tensor<4x!tt.ptr<i64>, #blocked>, tensor<4xi64, #blocked>) -> tensor<4xi64, #blocked>
    tt.return %a, %b : i64, tensor<4xi64, #blocked>
  }

  // CHECK-LABEL: @cas_to_cas_i64_tensor(
  tt.func @cas_to_cas_i64_tensor(%ptr: !tt.ptr<i64>, %next: tensor<4x!tt.ptr<i64>, #blocked>) -> (i64, tensor<4xi64, #blocked>) {
    %zero = arith.constant 0 : i64
    %zeros = arith.constant dense<0> : tensor<4xi64, #blocked>
    // CHECK: tt.atomic_cas{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 8 : i64
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: %{{.*}} = tt.atomic_cas{{.*}}test.buffer_offset = 0 : i64, test.buffer_size = 32 : i64
    %a = tt.atomic_cas acq_rel, gpu, %ptr, %zero, %zero {test.check_allocation} : (!tt.ptr<i64>, i64, i64) -> i64
    %b = tt.atomic_cas acq_rel, gpu, %next, %zeros, %zeros {test.check_allocation} : (tensor<4x!tt.ptr<i64>, #blocked>, tensor<4xi64, #blocked>, tensor<4xi64, #blocked>) -> tensor<4xi64, #blocked>
    tt.return %a, %b : i64, tensor<4xi64, #blocked>
  }
}
