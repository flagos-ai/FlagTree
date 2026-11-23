// RUN: triton-opt %s --tritongpu-global-scratch-memory-allocation --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' --canonicalize --cse | FileCheck %s

// In a 2x1x2 domain, CTAs 0/2 and 1/3 form the two groups. The unit
// middle axis must contribute zero to the counter index, regardless of CTA.
// CHECK-LABEL: llvm.func @unit_domain
// CHECK: %[[ONE:[a-zA-Z0-9_]+]] = llvm.mlir.constant(1 : i32)
// CHECK: llvm.urem %{{.*}}, %[[ONE]] : i32
// CHECK: llvm.inline_asm
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @unit_domain() {
    "tle.distributed_barrier"() <{group_kind = "grid_axis", group_rank = 1 : i32, group_shape = array<i32: 2>, group_axes = array<i32: 0>, group_domain_shape = array<i32: 2, 1, 2>}> : () -> ()
    tt.return
  }
}
