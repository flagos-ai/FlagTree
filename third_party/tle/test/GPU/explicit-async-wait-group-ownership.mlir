// RUN: triton-opt %s -tritongpu-pipeline | FileCheck %s

module attributes {ttg.target = "cuda:90", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A transport wait has no SSA dependency token. Its number belongs to the
  // explicit producer/consumer schedule, not the automatic pipeline analysis.
  // CHECK-LABEL: tt.func @explicit_empty_token_wait
  // CHECK: ttg.async_wait {num = 1 : i32, tle.explicit_async_wait_group}
  tt.func @explicit_empty_token_wait() {
    %c0 = ttg.async_commit_group
    %c1 = ttg.async_commit_group
    %w = ttg.async_wait {num = 1 : i32, tle.explicit_async_wait_group}
    tt.return
  }

  // CHECK-LABEL: tt.func @explicit_wait_inside_unmarked_loop
  // CHECK: scf.for
  // CHECK: ttg.async_wait {num = 2 : i32, tle.explicit_async_wait_group}
  tt.func @explicit_wait_inside_unmarked_loop() {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    scf.for %i = %zero to %two step %one {
      %w = ttg.async_wait {num = 2 : i32, tle.explicit_async_wait_group}
      scf.yield
    }
    tt.return
  }

  // The neighboring compiler-owned wait retains its existing inference.
  // CHECK-LABEL: tt.func @compiler_owned_wait
  // CHECK: ttg.async_wait {num = 0 : i32}
  tt.func @compiler_owned_wait() {
    %w = ttg.async_wait {num = 1 : i32}
    tt.return
  }

  // An unmarked wait must not consume and drop an explicit wait's ownership.
  // CHECK-LABEL: tt.func @adjacent_different_owners
  // CHECK: ttg.async_wait {num = 0 : i32}
  // CHECK: ttg.async_wait {num = 1 : i32, tle.explicit_async_wait_group}
  tt.func @adjacent_different_owners() {
    %implicit = ttg.async_wait {num = 0 : i32}
    %explicit = ttg.async_wait {num = 1 : i32, tle.explicit_async_wait_group}
    tt.return
  }
}
