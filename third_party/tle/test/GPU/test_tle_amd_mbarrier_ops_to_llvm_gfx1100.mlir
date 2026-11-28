// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1100 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=GFX1100

// The hardware named-barrier arrive intrinsic (ds.atomic.barrier.arrive.rtn.b64)
// only exists on gfx12.5+. On gfx1100 (RDNA3) init/wait are arch-neutral, but
// arrive must be emulated with a cmpxchg loop on the i64 LDS barrier word,
// keeping the same bit layout so the three ops still interoperate.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx1100", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1100-LABEL: init_barrier
  tt.func @init_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // GFX1100: %[[INIT_VAL1:.+]] = llvm.mlir.constant(4294967297 : i64) : i64
    // GFX1100: %[[ALLOC_PTR:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<3>, i32)>
    // GFX1100: llvm.store %[[INIT_VAL1]], %[[ALLOC_PTR]] : i64, !llvm.ptr<3>
    // GFX1100: rocdl.barrier
    amdg.init_barrier %alloc, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // GFX1100-LABEL: wait_barrier
  tt.func @wait_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %phase: i32) {
    // GFX1100: rocdl.s.sleep {{.*}}
    // GFX1100: llvm.load {{.*}} : !llvm.ptr<3> -> i64
    // GFX1100: llvm.icmp "ne" {{%arg1, %.*|%.*, %arg1}} : i32
    amdg.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // GFX1100-LABEL: arrive_barrier
  tt.func @arrive_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // No hardware barrier-arrive intrinsic on gfx1100.
    // GFX1100-NOT: ds.atomic.barrier.arrive
    // Software cmpxchg loop on the i64 LDS word at workgroup scope.
    // GFX1100: llvm.cmpxchg {{.*}} syncscope("workgroup") acq_rel monotonic : !llvm.ptr<3>, i64
    // GFX1100: llvm.extractvalue %{{[0-9]+}}[1] : !llvm.struct<(i64, i1)>
    // GFX1100: llvm.cond_br
    %prior_phase = amdg.arrive_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> i32
    tt.return
  }
}
