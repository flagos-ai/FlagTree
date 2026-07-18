// RUN: triton-opt %s -split-input-file --convert-triton-gpu-to-llvm=compute-capability=90 -reconcile-unrealized-casts | FileCheck %s

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: arrive_barrier_release_fence
  tt.func @arrive_barrier_release_fence(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>) {
    // CHECK: "membar.cta;\0A@$0 mbarrier.arrive.shared::cta.b64 _, [$1], 2;", "b,r"
    ttng.arrive_barrier %alloc, 2 {release_fence = true} : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }

  // CHECK-LABEL: participant_arrive_barrier_release_fence
  tt.func @participant_arrive_barrier_release_fence(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>) {
    // CHECK: "@$0 membar.cta;\0A@$0 mbarrier.arrive.shared::cta.b64 _, [$1];", "b,r"
    ttng.arrive_barrier %alloc, 64 {participant_arrive = true, release_fence = true} : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}
