// RUN: triton-opt %s --allocate-shared-memory --test-print-membar -verify-each --allow-unregistered-dialect | FileCheck %s
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  tt.func private @callee() { tt.return }
  // CHECK-LABEL: tt.func public @ws_shared_read
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_shared_read(%arena: !ttg.memdesc<128xi32, #shared, #smem, mutable>) {
    ttg.warp_specialize() default {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    %v = ttg.local_load %arena : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #blocked>
    gpu.barrier
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_shared_write
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_shared_write(%arena: !ttg.memdesc<128xi32, #shared, #smem, mutable>, %v: tensor<128xi32, #blocked>) {
    ttg.warp_specialize() default {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    ttg.local_store %v, %arena : tensor<128xi32, #blocked> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    gpu.barrier
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_global_write
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_global_write(%ptr: !tt.ptr<i32>, %v: i32) {
    ttg.warp_specialize() default {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    tt.store %ptr, %v : !tt.ptr<i32>
    gpu.barrier
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_atomic
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_atomic(%ptr: !tt.ptr<i64>, %v: i64, %mask: i1) {
    ttg.warp_specialize() default {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    %a = tt.atomic_rmw add, relaxed, sys, %ptr, %v, %mask : (!tt.ptr<i64>, i64, i1) -> i64
    gpu.barrier
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_call
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_call() {
    ttg.warp_specialize() default {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    tt.call @callee() : () -> ()
    gpu.barrier
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_unknown
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_unknown() {
    ttg.warp_specialize() default {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    "test.unknown"() : () -> ()
    gpu.barrier
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_async_commit
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_async_commit() {
    ttg.warp_specialize() default {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    ttg.async_commit_group
    gpu.barrier
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_conditional
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_conditional(%pred: i1) {
    ttg.warp_specialize() default {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    cf.cond_br %pred, ^left, ^right
  ^left:
    gpu.barrier
    ttg.warp_yield
  ^right:
    gpu.barrier
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_cycle
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_cycle() {
    ttg.warp_specialize() default {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    cf.br ^loop
  ^loop:
    cf.br ^loop
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_no_barrier_exit
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_no_barrier_exit() {
    ttg.warp_specialize() default {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    cf.br ^exit
  ^exit:
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_non_cta_fence
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_non_cta_fence() {
    ttg.warp_specialize() default {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    ttng.fence_async_shared {bCluster = false}
    gpu.barrier
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_nested_region
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_nested_region() {
    ttg.warp_specialize() default {
    scf.execute_region {
      ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
      %v = arith.constant 1 : i32
      gpu.barrier
      scf.yield
    }
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_native_tma_store_wait
  // CHECK: ttng.async_tma_store_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_native_tma_store_wait() {
    ttg.warp_specialize() default {
    ttng.async_tma_store_wait {pendings = 0 : i32}
    cf.br ^merge
  ^merge:
    gpu.barrier
    ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_unmarked_merge
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_unmarked_merge() {
    ttg.warp_specialize() default {
      ttg.async_wait {num = 0 : i32}
      cf.br ^merge
    ^merge:
      gpu.barrier
      ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_join_outer_barrier
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_join_outer_barrier() {
    ttg.warp_specialize() default {
      ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
      cf.br ^exit
    ^exit:
      ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_other_partition_barrier
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_other_partition_barrier() {
    ttg.warp_specialize() default {
      ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
      cf.br ^exit
    ^exit:
      ttg.warp_yield
    } partition0() num_warps(1) {
      gpu.barrier
      ttg.warp_return
    } : () -> ()
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @ws_region_op_boundary
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @ws_region_op_boundary() {
    ttg.warp_specialize() default {
      ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
      scf.execute_region {
        gpu.barrier
        scf.yield
      }
      gpu.barrier
      ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    gpu.barrier
    tt.return
  }
}
