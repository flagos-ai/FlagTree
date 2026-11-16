// RUN: triton-opt %s --convert-scf-to-cf --allocate-shared-memory --test-print-membar -verify-each | FileCheck %s
// Reviewer CPU-only cause probe; unchanged compiler, no native launch.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  // CHECK-LABEL: tt.func public @owned_merge
  // CHECK: ttg.async_wait {num = 2
  // CHECK-NEXT: cf.br
  // CHECK: ttg.async_wait {num = 1
  // CHECK-NEXT: cf.br
  // CHECK: ttg.async_wait {num = 0
  // CHECK-NEXT: cf.br
  // CHECK: gpu.barrier
  // CHECK-NEXT: tt.return
  tt.func public @owned_merge() {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %a = arith.cmpi eq, %pid, %c0 : i32
    %b = arith.cmpi eq, %pid, %c1 : i32
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
    tt.return
  }
  // CHECK-LABEL: tt.func public @owned_adjacent
  // CHECK: ttg.async_wait
  // CHECK-NEXT: gpu.barrier
  // CHECK-NEXT: tt.return
  tt.func public @owned_adjacent() {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @unmarked_merge
  // CHECK: ttg.async_wait
  // CHECK-NEXT: ttg.local_barrier
  tt.func public @unmarked_merge() {
    %pid = tt.get_program_id x : i32
    %c0 = arith.constant 0 : i32
    %a = arith.cmpi eq, %pid, %c0 : i32
    scf.if %a {
      ttg.async_wait {num = 1 : i32}
    } else {
      ttg.async_wait {num = 0 : i32}
    }
    gpu.barrier
    tt.return
  }
  // CHECK-LABEL: tt.func public @owned_pure_chain_local
  // CHECK: ttg.async_wait
  // CHECK-NEXT: arith.constant
  // CHECK-NOT: ttg.local_barrier
  // CHECK: cf.br
  // CHECK: ttg.local_barrier
  // CHECK-NEXT: tt.return
  tt.func public @owned_pure_chain_local() {
    ttg.async_wait {num = 0 : i32, tle.explicit_async_wait_group}
    %v = arith.constant 1 : i32
    cf.br ^forward
  ^forward:
    %w = arith.addi %v, %v : i32
    cf.br ^merge
  ^merge:
    ttg.local_barrier
    tt.return
  }
}
