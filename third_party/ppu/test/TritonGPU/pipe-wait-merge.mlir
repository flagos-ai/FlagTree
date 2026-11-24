// RUN: triton-opt %s -tritongpu-pipeline | FileCheck %s

// The pipeline pass merges adjacent async waits. A wait whose distance a pipe
// lowering chose (tle.pipe_wait) is not merged into a neighbour: a later
// update of the wait counts cannot reconstruct its distance.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @pipe_wait_stays
  tt.func @pipe_wait_stays() {
    %group = ttg.async_commit_group
    // CHECK: ttg.async_wait
    // CHECK-NEXT: ttg.async_wait {{.*}}tle.pipe_wait
    %wait = ttg.async_wait %group {num = 0 : i32}
    %pipe_wait = ttg.async_wait {num = 0 : i32, tle.pipe_wait}
    tt.return
  }

  // Waits without the marker still merge.
  // CHECK-LABEL: tt.func @plain_waits_merge
  tt.func @plain_waits_merge() {
    %group = ttg.async_commit_group
    // CHECK: ttg.async_wait
    // CHECK-NOT: ttg.async_wait
    %first = ttg.async_wait %group {num = 0 : i32}
    %second = ttg.async_wait {num = 0 : i32}
    tt.return
  }
}
