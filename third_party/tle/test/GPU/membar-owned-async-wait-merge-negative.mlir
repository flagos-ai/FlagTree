// RUN: triton-opt %s --allocate-shared-memory --test-print-membar -verify-each --allow-unregistered-dialect | FileCheck %s
// Each non-proven path keeps the original immediate CTA synchronization.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func private @callee() { tt.return }
  // CHECK-LABEL: tt.func public @shared_read
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @shared_read(%arena: !ttg.memdesc<128xi32, #shared, #smem, mutable>) {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    %v = ttg.local_load %arena : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #blocked>
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @shared_write
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @shared_write(%arena: !ttg.memdesc<128xi32, #shared, #smem, mutable>, %v: tensor<128xi32, #blocked>) {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    ttg.local_store %v, %arena : tensor<128xi32, #blocked> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @global_write
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @global_write(%ptr: !tt.ptr<i32>, %v: i32) {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    tt.store %ptr, %v : !tt.ptr<i32>
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @atomic
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @atomic(%ptr: !tt.ptr<i64>, %v: i64, %mask: i1) {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    %a = tt.atomic_rmw add, relaxed, sys, %ptr, %v, %mask : (!tt.ptr<i64>, i64, i1) -> i64
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @call
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @call() {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    tt.call @callee() : () -> ()
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @unknown
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @unknown() {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    "test.unknown"() : () -> ()
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @async_commit
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @async_commit() {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    ttg.async_commit_group
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @conditional
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @conditional(%pred: i1) {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    cf.cond_br %pred, ^left, ^right
  ^left:
    gpu.barrier
    tt.return
  ^right:
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @cycle
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @cycle() {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    cf.br ^loop
  ^loop:
    cf.br ^loop
  }
  // CHECK-LABEL: tt.func public @no_barrier_exit
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @no_barrier_exit() {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    cf.br ^exit
  ^exit:
    tt.return
  }
  // CHECK-LABEL: tt.func public @non_cta_fence
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @non_cta_fence() {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    ttng.fence_async_shared {bCluster = false}
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @nested_region
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @nested_region() {
    scf.execute_region {
      ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
      %v = arith.constant 1 : i32
      gpu.barrier
      scf.yield
    }
    tt.return
  }
  // CHECK-LABEL: tt.func public @native_tma_store_wait
  // CHECK: ttng.async_tma_store_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @native_tma_store_wait() {
    ttng.async_tma_store_wait {pendings = 0 : i32}
    cf.br ^merge
  ^merge:
    gpu.barrier
    tt.return
  }
}
