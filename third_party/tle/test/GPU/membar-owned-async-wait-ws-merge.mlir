// RUN: triton-opt %s --convert-scf-to-cf --allocate-shared-memory --test-print-membar -verify-each | FileCheck %s
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: tt.func public @owned_ws_default_merge
  // CHECK: ttg.async_wait {num = 2
  // CHECK-NEXT: cf.br
  // CHECK: ttg.async_wait {num = 1
  // CHECK-NEXT: cf.br
  // CHECK: ttg.async_wait {num = 0
  // CHECK-NEXT: cf.br
  // CHECK: gpu.barrier
  // CHECK-NEXT: ttg.warp_yield
  tt.func public @owned_ws_default_merge() {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %a = arith.cmpi eq, %pid, %c0 : i32
    %b = arith.cmpi eq, %pid, %c1 : i32
    ttg.warp_specialize() default {
      scf.if %a {
        ttg.async_wait {num = 2 : i32, tle.explicit_async_wait_group}
      } else {
        scf.if %b {
          ttg.async_wait {num = 1 : i32, tle.explicit_async_wait_group}
        } else {
          ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
        }
      }
      gpu.barrier
      ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @owned_ws_worker_merge
  // CHECK: ttg.async_wait
  // CHECK-NEXT: cf.br
  // CHECK: ttg.local_barrier
  // CHECK-NEXT: ttg.warp_return
  tt.func public @owned_ws_worker_merge() {
    ttg.warp_specialize() default {
      ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
      cf.br ^merge
    ^merge:
      ttg.local_barrier
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
