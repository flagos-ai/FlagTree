// RUN: triton-opt %s -tritongpu-pipeline | FileCheck %s

module attributes {ttg.target = "cuda:90", "ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @preserve_explicit_wait
  // CHECK: scf.for
  // CHECK: ttg.async_wait {{.*}} {num = 1 : i32}
  tt.func @preserve_explicit_wait() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %tok = ttg.async_commit_group
    scf.for %i = %c0 to %c2 step %c1 {
      %w = ttg.async_wait %tok {num = 1 : i32}
      scf.yield
    } {tle.explicit_tile_style_pipeline = 1 : i32}
    tt.return
  }
}
