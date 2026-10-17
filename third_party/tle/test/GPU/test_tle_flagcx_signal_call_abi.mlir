// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=90 ptx-version=80}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=80})' | FileCheck %s

// Signal and wait helpers share the same generic-pointer C ABI as rank and
// barrier helpers. Check all branches without requiring a network runtime.
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-DAG: llvm.func @flagcxDevSignalInc(!llvm.ptr, i32, i32, i32, i32, i32, i32)
  // CHECK-DAG: llvm.func @flagcxDevSignalAdd(!llvm.ptr, i32, i32, i32, i64, i32, i32, i32)
  // CHECK-DAG: llvm.func @flagcxDevWaitSignal(!llvm.ptr, i32, i64, i32, i32, i32, i32)
  // CHECK-DAG: llvm.func @flagcxDevWaitSignalMeetShadow(!llvm.ptr, i32, i32, i32, i32, i32)
  // CHECK-DAG: llvm.func @flagcxDevWaitCounter(!llvm.ptr, i32, i64, i32, i32, i32, i32)
  // CHECK-LABEL: llvm.func @kernel
  // CHECK: llvm.call @flagcxDevSignalInc({{.*}}) : (!llvm.ptr, i32, i32, i32, i32, i32, i32) -> ()
  // CHECK: llvm.call @flagcxDevSignalAdd({{.*}}) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32) -> ()
  // CHECK: llvm.call @flagcxDevWaitSignal({{.*}}) : (!llvm.ptr, i32, i64, i32, i32, i32, i32) -> ()
  // CHECK: llvm.call @flagcxDevWaitSignalMeetShadow({{.*}}) : (!llvm.ptr, i32, i32, i32, i32, i32) -> ()
  // CHECK: llvm.call @flagcxDevWaitCounter({{.*}}) : (!llvm.ptr, i32, i64, i32, i32, i32, i32) -> ()
  tt.func public @kernel(%comm: i64, %peer: i32, %slot: i32, %value: i64) {
    "tle.flagcx signal"(%comm, %peer, %slot) {signal_op = 0 : i32, team_kind = 0 : i32, coop_kind = 2 : i32, context_idx = 0 : i32} : (i64, i32, i32) -> ()
    "tle.flagcx signal"(%comm, %peer, %slot, %value) {signal_op = 1 : i32, team_kind = 2 : i32, coop_kind = 1 : i32, context_idx = 0 : i32} : (i64, i32, i32, i64) -> ()
    "tle.flagcx signal_wait"(%comm, %slot, %value) {wait_kind = 0 : i32, coop_kind = 2 : i32, context_idx = 0 : i32} : (i64, i32, i64) -> ()
    "tle.flagcx signal_wait"(%comm, %slot) {wait_kind = 1 : i32, coop_kind = 1 : i32, context_idx = 0 : i32} : (i64, i32) -> ()
    "tle.flagcx signal_wait"(%comm, %slot, %value) {wait_kind = 2 : i32, coop_kind = 0 : i32, context_idx = 0 : i32} : (i64, i32, i64) -> ()
    tt.return
  }
}
