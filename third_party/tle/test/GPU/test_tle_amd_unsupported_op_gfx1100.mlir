// RUN: not triton-opt %s -convert-triton-amdgpu-to-llvm=arch=gfx1100 2>&1 | FileCheck %s

// F1 capability gate (FlagMega-on-Radeon Phase 0): a TLE op that does not yet
// have an AMD/RDNA lowering must produce a clear, actionable diagnostic naming
// the op and the arch, instead of a downstream crash or a generic
// "failed to legalize" error. Update this once the op gains an RDNA lowering.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @unsupported_tle_op() {
    // CHECK: 'tle.barrier.alloc' op has no AMD/RDNA lowering yet (arch gfx1100)
    %bars = tle.barrier.alloc {arrive_count = 2 : i32, init_polarity = 0 : i32, num_barriers = 2 : i32} : !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    tt.return
  }
}
