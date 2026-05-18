// RUN: triton-opt %s -split-input-file -verify-diagnostics

#shared0 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_participant_arrive_without_release_fence(%alloc: !ttg.memdesc<1xi64, #shared0, #smem>) {
    // expected-error @+1 {{participant_arrive requires release_fence}}
    ttng.arrive_barrier %alloc, 64 {participant_arrive = true} : !ttg.memdesc<1xi64, #shared0, #smem>
    tt.return
  }
}
