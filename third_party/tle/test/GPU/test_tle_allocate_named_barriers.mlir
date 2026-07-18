// RUN: triton-opt %s -triton-tle-allocate-named-barriers -split-input-file | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @remap_virtual_ids_without_warpspec
  tt.func @remap_virtual_ids_without_warpspec() {
    %v0 = arith.constant 16 : i32
    %v1 = arith.constant 17 : i32
    %threads = arith.constant 256 : i32

    // CHECK: %[[ID0:.+]] = arith.constant 1 : i32
    // CHECK: ttng.wait_barrier_named %[[ID0]], {{.*}} : i32, i32
    ttng.wait_barrier_named %v0, %threads : i32, i32

    // CHECK: %[[ID1:.+]] = arith.constant 2 : i32
    // CHECK: ttng.arrive_barrier_named %[[ID1]], {{.*}} : i32, i32
    ttng.arrive_barrier_named %v1, %threads : i32, i32

    // CHECK: %[[ID0_AGAIN:.+]] = arith.constant 1 : i32
    // CHECK: ttng.arrive_barrier_named %[[ID0_AGAIN]], {{.*}} : i32, i32
    ttng.arrive_barrier_named %v0, %threads : i32, i32
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @remap_virtual_ids_with_warpspec
  tt.func @remap_virtual_ids_with_warpspec() {
    ttg.warp_specialize() attributes {requestedRegisters = array<i32: 128, 128>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %v0 = arith.constant 16 : i32
      %threads = arith.constant 256 : i32
      // CHECK: %[[ID4:.+]] = arith.constant 4 : i32
      // CHECK: ttng.wait_barrier_named %[[ID4]], {{.*}} : i32, i32
      ttng.wait_barrier_named %v0, %threads : i32, i32
      ttg.warp_return
    }
    partition1() num_warps(4) {
      %v1 = arith.constant 17 : i32
      %threads = arith.constant 256 : i32
      // CHECK: %[[ID5:.+]] = arith.constant 5 : i32
      // CHECK: ttng.arrive_barrier_named %[[ID5]], {{.*}} : i32, i32
      ttng.arrive_barrier_named %v1, %threads : i32, i32
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @preserve_existing_physical_ids_with_warpspec
  tt.func @preserve_existing_physical_ids_with_warpspec() {
    ttg.warp_specialize() attributes {requestedRegisters = array<i32: 128, 128>}
    default {
      ttg.warp_yield
    }
    partition0() num_warps(4) {
      %physical0 = arith.constant 0 : i32
      %threads = arith.constant 256 : i32
      // CHECK: %[[PHYSICAL0:.+]] = arith.constant 0 : i32
      // CHECK: ttng.wait_barrier_named %[[PHYSICAL0]], {{.*}} : i32, i32
      ttng.wait_barrier_named %physical0, %threads : i32, i32
      ttg.warp_return
    }
    partition1() num_warps(4) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
