// RUN: triton-opt %s -split-input-file -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=90 ptx-version=80},tritongpu-global-scratch-memory-allocation,convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=80},cse)' | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Each CTA uses its own slot; the runtime allocates at least 17 slots.
  // Both rendezvous must survive CSE, including when the index is identical.
  // CHECK-LABEL: llvm.func @indexed(
  // CHECK-SAME: %[[INDEX:[a-zA-Z0-9_]+]]: i32
  // CHECK: llvm.call @flagcxIntraBarrierSyncS({{.*}}, %[[INDEX]], {{.*}}) : (!llvm.ptr, i32, i32, i1, i32) -> ()
  // CHECK: llvm.call @flagcxIntraBarrierSyncS({{.*}}, %[[INDEX]], {{.*}}) : (!llvm.ptr, i32, i32, i1, i32) -> ()
  // CHECK: llvm.return
  tt.func public @indexed(%comm: i64, %slot: i32) {
    "tle.device intra barrier"(%comm, %slot) <{barrier_type = "sync", coop_kind = 2 : i32, multimem = false, order = 3 : i32, barrier_count = 17 : i32}> : (i64, i32) -> ()
    "tle.device intra barrier"(%comm, %slot) <{barrier_type = "sync", coop_kind = 2 : i32, multimem = false, order = 3 : i32, barrier_count = 17 : i32}> : (i64, i32) -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @legacy_static
  // CHECK: llvm.call @flagcxIntraBarrierSyncS
  // CHECK: llvm.return
  tt.func public @legacy_static(%comm: i64) {
    "tle.device intra barrier"(%comm) <{barrier_type = "sync", coop_kind = 2 : i32, index = 0 : i32, multimem = false, order = 3 : i32}> : (i64) -> ()
    tt.return
  }
}
