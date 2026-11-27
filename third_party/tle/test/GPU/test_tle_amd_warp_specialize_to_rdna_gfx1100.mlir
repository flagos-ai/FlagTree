// RUN: triton-opt %s -split-input-file -tritonamdgpu-convert-warp-specialize | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritonamdgpu-convert-warp-specialize -allocate-shared-memory -convert-triton-amdgpu-to-llvm=arch=gfx1100 | FileCheck %s --check-prefix=LLVM

// FlagMega-on-Radeon P1-B: ttg.warp_specialize lowers to a static wave-id
// partition — wave id = workitem.id.x / warpSize, a cf branch chain dispatches
// the default warp group and each partition's wave range, and a CTA-wide
// gpu.barrier rejoins. No ttg.warp_specialize / warp_yield / warp_return survive.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // The worker waves sit above the 4 default waves, so the CTA must launch
  // 4 + 4 = 8 waves; the pass records that for the backend launch count.
  // CHECK: "ttg.total-num-warps" = 8 : i32
  // CHECK-LABEL: @ws_basic
  // CHECK: %[[TID:.*]] = rocdl.workitem.id.x
  // CHECK: arith.divui %[[TID]]
  // CHECK: gpu.barrier
  // CHECK: cf.cond_br
  // CHECK: arith.addi
  // CHECK: gpu.barrier
  // CHECK-NOT: ttg.warp_specialize
  // CHECK-NOT: ttg.warp_yield
  // CHECK-NOT: ttg.warp_return

  // LLVM-LABEL: llvm.func @ws_basic
  // LLVM: rocdl.workitem.id.x
  // LLVM-NOT: ttg.warp_specialize
  tt.func public @ws_basic(%arg0: i32) {
    ttg.warp_specialize(%arg0) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%a: i32) num_warps(4) {
      %x = arith.addi %a, %a : i32
      ttg.warp_return
    } : (i32) -> ()
    tt.return
  }
}

// -----

// Two partitions at wave ranges [4,8) and [8,10); start ids assigned here.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // Waves span [0,4) default, [4,8), [8,10) -> launch 10 waves.
  // CHECK: "ttg.total-num-warps" = 10 : i32
  // CHECK-LABEL: @ws_two_partitions
  // CHECK: rocdl.workitem.id.x
  // CHECK-COUNT-3: cf.cond_br
  // CHECK-NOT: ttg.warp_specialize
  tt.func public @ws_two_partitions(%arg0: i32, %arg1: i32) {
    ttg.warp_specialize(%arg0, %arg1) attributes {warpGroupStartIds = array<i32: 4, 8>}
    default {
      ttg.warp_yield
    }
    partition0(%a: i32, %b: i32) num_warps(4) {
      %x = arith.addi %a, %b : i32
      ttg.warp_return
    }
    partition1(%a: i32, %b: i32) num_warps(2) {
      %y = arith.subi %a, %b : i32
      ttg.warp_return
    } : (i32, i32) -> ()
    tt.return
  }
}

// -----

// A scalar warp_yield result is broadcast to all waves through an LDS slot:
// the default waves store it at warp_yield, all waves load it after rejoin.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: "ttg.total-num-warps" = 8 : i32
  // CHECK-LABEL: @ws_yields_scalar
  // CHECK: %[[SLOT:.*]] = ttg.local_alloc : () -> !ttg.memdesc<1xi32
  // default stores the yielded value:
  // CHECK: ttg.local_store %{{.*}}, %[[SLOT]]
  // all waves reload after the rejoin barrier:
  // CHECK: gpu.barrier
  // CHECK: ttg.local_load %[[SLOT]]
  // CHECK: tt.unsplat
  // CHECK-NOT: ttg.warp_specialize
  tt.func public @ws_yields_scalar(%arg0: i32, %ptr: !tt.ptr<i32>) {
    %r = ttg.warp_specialize(%arg0) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      %c = arith.addi %arg0, %arg0 : i32
      ttg.warp_yield %c : i32
    }
    partition0(%a: i32) num_warps(4) {
      ttg.warp_return
    } : (i32) -> i32
    tt.store %ptr, %r : !tt.ptr<i32>
    tt.return
  }
}
