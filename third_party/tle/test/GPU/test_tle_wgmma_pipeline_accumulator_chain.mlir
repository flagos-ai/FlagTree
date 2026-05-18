// RUN: triton-opt %s -split-input-file -tritongpu-assign-latencies -tritongpu-schedule-loops -tritongpu-pipeline -canonicalize | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @defer_wait_for_same_iteration_wgmma_c_chain
  tt.func @defer_wait_for_same_iteration_wgmma_c_chain(
      %a0: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b0: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %a1: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b1: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %b0_view = tle.memdesc_wgmma_view %b0 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    %b1_view = tle.memdesc_wgmma_view %b1 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot
      %dot0 = ttng.warp_group_dot %a0, %b0_view, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot {{.*}}, %[[DOT0]]
      // CHECK-SAME: tle.wgmma_accumulator_chain_c
      %dot1 = ttng.warp_group_dot %a1, %b1_view, %dot0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT1]]
      // CHECK-SAME: {pendings = 0 : i32}
      // CHECK: tt.store %{{.*}}, %[[WAIT]]#0
      tt.store %out, %dot1 : tensor<64x64x!tt.ptr<f32>, #mma>
    }
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @branch_release_wait_does_not_cover_loop_result
  tt.func @branch_release_wait_does_not_cover_loop_result(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %slot: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    // CHECK: %[[RES:.+]] = scf.for
    %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
      %dot = ttng.warp_group_dot %a, %slot, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[COND:.+]] = arith.cmpi
      %cond = arith.cmpi slt, %iv, %c8 : index
      // CHECK-NEXT: scf.if %[[COND]]
      scf.if %cond {
        // CHECK: %[[WAIT_THEN:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]
        // CHECK-SAME: {pendings = 0 : i32}
        // CHECK: nvws.consumer_release
        nvws.consumer_release %token, %idx, %slot {release_count = 128 : i32} : tensor<2x!nvws.token>, i32, !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      } else {
        // CHECK: %[[WAIT_ELSE:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]
        // CHECK-SAME: {pendings = 0 : i32}
        // CHECK: nvws.consumer_release
        nvws.consumer_release %token, %idx, %slot {release_count = 128 : i32} : tensor<2x!nvws.token>, i32, !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      }
      // CHECK: scf.yield %[[DOT]]
      scf.yield %dot : tensor<64x64xf32, #mma>
    }
    // CHECK: tt.store %{{.*}}, %[[RES]]
    tt.store %out, %res : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @defer_wait_for_three_dot_wgmma_c_chain
  tt.func @defer_wait_for_three_dot_wgmma_c_chain(
      %a0: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b0: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %a1: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b1: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %a2: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b2: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %b0_view = tle.memdesc_wgmma_view %b0 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    %b1_view = tle.memdesc_wgmma_view %b1 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    %b2_view = tle.memdesc_wgmma_view %b2 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot
      %dot0 = ttng.warp_group_dot %a0, %b0_view, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot {{.*}}, %[[DOT0]]
      // CHECK-SAME: tle.wgmma_accumulator_chain_c
      %dot1 = ttng.warp_group_dot %a1, %b1_view, %dot0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: %[[DOT2:.+]] = ttng.warp_group_dot {{.*}}, %[[DOT1]]
      // CHECK-SAME: tle.wgmma_accumulator_chain_c
      %dot2 = ttng.warp_group_dot %a2, %b2_view, %dot1 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT2]]
      // CHECK-SAME: {pendings = 0 : i32}
      // CHECK: tt.store %{{.*}}, %[[WAIT]]#0
      tt.store %out, %dot2 : tensor<64x64x!tt.ptr<f32>, #mma>
    }
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @shared_operand_fence_forces_wait_before_next_chain_c
  tt.func @shared_operand_fence_forces_wait_before_next_chain_c(
      %a0: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b0: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %a1: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b1: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %b0_view = tle.memdesc_wgmma_view %b0 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot
      %dot0 = ttng.warp_group_dot %a0, %b0_view, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK: %[[B1_VIEW:.+]] = tle.memdesc_wgmma_view
      %b1_view = tle.memdesc_wgmma_view %b1 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
      // CHECK-NEXT: tle.wgmma_shared_operand_fence %[[B1_VIEW]]
      tle.wgmma_shared_operand_fence %b1_view {bCluster = false} : !ttg.memdesc<64x64xbf16, #shared1, #smem>
      // CHECK-NEXT: %[[WAIT0:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT0]]
      // CHECK-SAME: {pendings = 0 : i32}
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot {{.*}}, %[[WAIT0]]#0 {inputPrecision = 0 : i32, isAsync = true, tle.explicit_wgmma_commit}
      %dot1 = ttng.warp_group_dot %a1, %b1_view, %dot0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT1]]
      // CHECK-SAME: {pendings = 0 : i32}
      tt.store %out, %dot1 : tensor<64x64x!tt.ptr<f32>, #mma>
    }
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @chain_c_across_target_operand_local_load
  tt.func @chain_c_across_target_operand_local_load(
      %a0: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b0: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %a1: !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>,
      %b1: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %b0_view = tle.memdesc_wgmma_view %b0 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    %b1_view = tle.memdesc_wgmma_view %b1 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot
      %dot0 = ttng.warp_group_dot %a0, %b0_view, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK: %[[A1:.+]] = ttg.local_load
      %a1_load = ttg.local_load %a1 : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> tensor<64x64xbf16, #dot>
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot %[[A1]], {{.*}}, %[[DOT0]]
      // CHECK-SAME: tle.wgmma_accumulator_chain_c
      %dot1 = ttng.warp_group_dot %a1_load, %b1_view, %dot0 {inputPrecision = 0 : i32} : tensor<64x64xbf16, #dot> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT1]]
      // CHECK-SAME: {pendings = 0 : i32}
      tt.store %out, %dot1 : tensor<64x64x!tt.ptr<f32>, #mma>
    }
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @wait_c_across_wait_barrier_without_release
  tt.func @wait_c_across_wait_barrier_without_release(
      %a0: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b0: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %a1: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b1: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0_i32 = arith.constant 0 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %b0_view = tle.memdesc_wgmma_view %b0 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    %b1_view = tle.memdesc_wgmma_view %b1 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot
      %dot0 = ttng.warp_group_dot %a0, %b0_view, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: ttng.wait_barrier
      ttng.wait_barrier %barrier, %c0_i32 : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      // CHECK-NEXT: %[[WAIT0:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT0]]
      // CHECK-SAME: {pendings = 0 : i32}
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot {{.*}}, %[[WAIT0]]#0 {inputPrecision = 0 : i32, isAsync = true, tle.explicit_wgmma_commit}
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: ttng.warp_group_dot_wait %[[DOT1]]
      %dot1 = ttng.warp_group_dot %a1, %b1_view, %dot0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      tt.store %out, %dot1 : tensor<64x64x!tt.ptr<f32>, #mma>
    }
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @arrive_barrier_released_slot_uses_modulo_alias
  tt.func @arrive_barrier_released_slot_uses_modulo_alias(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %slots: !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable>,
      %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %out0: tensor<64x64x!tt.ptr<f32>, #mma>,
      %out1: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      %iv_i32 = arith.index_cast %iv : index to i32
      %ck0 = arith.muli %iv_i32, %c2_i32 : i32
      %ck1 = arith.addi %ck0, %c1_i32 : i32
      %next = arith.addi %ck0, %c2_i32 : i32
      %release_idx = arith.remsi %ck1, %c2_i32 : i32
      %next_idx = arith.remsi %next, %c2_i32 : i32
      %release_slot = ttg.memdesc_index %slots[%release_idx] : !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      %next_slot = ttg.memdesc_index %slots[%next_idx] : !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot %{{.*}}, %[[RELEASE_SLOT:.+]], %{{.*}}
      %dot0 = ttng.warp_group_dot %a, %release_slot, %acc0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot %{{.*}}, %{{.*}}, %{{.*}}
      %dot1 = ttng.warp_group_dot %a, %next_slot, %acc1 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait {{.*}}%[[DOT0]]
      // CHECK-SAME: {pendings = 1 : i32}
      // CHECK-NEXT: ttng.arrive_barrier %{{.*}}, 128 released[{{.*}}] ({{.*}})
      ttng.arrive_barrier %barrier, 128 released[%release_idx](%slots) : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, i32, !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable>
      scf.yield %dot0, %dot1 : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    }
    tt.store %out0, %res#0 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.store %out1, %res#1 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @staged_wgmma_operand_uses_source_slot_for_release_alias
  tt.func @staged_wgmma_operand_uses_source_slot_for_release_alias(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %other: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %out0: tensor<64x64x!tt.ptr<f32>, #mma>,
      %out1: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %slots = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable>
    %slot = ttg.memdesc_index %slots[%idx] : !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
    %loaded = ttg.local_load %slot : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xbf16, #blocked>
    %staged = ttg.local_alloc %loaded : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot %{{.*}}, %[[STAGED:.+]], %{{.*}}
      %dot = ttng.warp_group_dot %a, %staged, %acc0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[NEXT:.+]] = ttng.warp_group_dot
      %next = ttng.warp_group_dot %a, %other, %acc1 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]
      // CHECK-SAME: {pendings = 1 : i32}
      // CHECK-NEXT: ttng.arrive_barrier %{{.*}}, 128 released[{{.*}}] ({{.*}})
      ttng.arrive_barrier %barrier, 128 released[%idx](%slots) : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, i32, !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable>
      scf.yield %dot, %next : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    }
    tt.store %out0, %res#0 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.store %out1, %res#1 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @foldable_wgmma_a_local_load_release_waits_for_source_slot
  tt.func @foldable_wgmma_a_local_load_release_waits_for_source_slot(
      %a_other: !ttg.memdesc<64x16xbf16, #shared, #smem>,
      %b0: !ttg.memdesc<16x64xbf16, #shared1, #smem, mutable>,
      %b1: !ttg.memdesc<16x64xbf16, #shared1, #smem, mutable>,
      %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %out0: tensor<64x64x!tt.ptr<f32>, #mma>,
      %out1: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %a_slots = ttg.local_alloc : () -> !ttg.memdesc<2x64x16xbf16, #shared, #smem, mutable>
    %res:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      // CHECK: %[[A_SLOT:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}]
      %a_slot = ttg.memdesc_index %a_slots[%idx] : !ttg.memdesc<2x64x16xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xbf16, #shared, #smem, mutable>
      // CHECK-NEXT: %[[A_LOAD:.+]] = ttg.local_load %[[A_SLOT]]
      %a_load = ttg.local_load %a_slot : !ttg.memdesc<64x16xbf16, #shared, #smem, mutable> -> tensor<64x16xbf16, #dot>
      // CHECK-NEXT: %[[DOT0:.+]] = ttng.warp_group_dot %[[A_LOAD]]
      %dot0 = ttng.warp_group_dot %a_load, %b0, %acc0 {inputPrecision = 0 : i32} : tensor<64x16xbf16, #dot> * !ttg.memdesc<16x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot
      %dot1 = ttng.warp_group_dot %a_other, %b1, %acc1 {inputPrecision = 0 : i32} : !ttg.memdesc<64x16xbf16, #shared, #smem> * !ttg.memdesc<16x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait {{.*}}%[[DOT0]]
      // CHECK-SAME: {pendings = 1 : i32}
      // CHECK-NEXT: ttng.arrive_barrier %{{.*}}, 128 released[{{.*}}] ({{.*}})
      ttng.arrive_barrier %barrier, 128 released[%idx](%a_slots) : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>, i32, !ttg.memdesc<2x64x16xbf16, #shared, #smem, mutable>
      scf.yield %dot0, %dot1 : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    }
    tt.store %out0, %res#0 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.store %out1, %res#1 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @redundant_wgmma_waits_without_intervening_dot_are_merged
  tt.func @redundant_wgmma_waits_without_intervening_dot_are_merged(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
      %dot = ttng.warp_group_dot %a, %b, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]
      // CHECK-SAME: {pendings = 0 : i32}
      %wait0:2 = ttng.warp_group_dot_wait %dot, %b {pendings = 0 : i32} : tensor<64x64xf32, #mma>, !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      %wait1:2 = ttng.warp_group_dot_wait %wait0#0, %wait0#1 {pendings = 0 : i32} : tensor<64x64xf32, #mma>, !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      // CHECK-NOT: ttng.warp_group_dot_wait
      // CHECK: nvws.consumer_release
      nvws.consumer_release %token, %idx, %wait1#1 {release_count = 128 : i32} : tensor<2x!nvws.token>, i32, !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      tt.store %out, %wait1#0 : tensor<64x64x!tt.ptr<f32>, #mma>
    }
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @nested_release_waits_after_branch_local_wgmma
  tt.func @nested_release_waits_after_branch_local_wgmma(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %slot: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %other: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out0: tensor<64x64x!tt.ptr<f32>, #mma>,
      %out1: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %res:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot %{{.*}}, %[[SLOT:.+]], %{{.*}}
      %dot = ttng.warp_group_dot %a, %slot, %acc0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[COND:.+]] = arith.cmpi
      %cond = arith.cmpi slt, %iv, %c8 : index
      // CHECK-NEXT: scf.if %[[COND]]
      scf.if %cond {
        // CHECK: %[[NEXT:.+]] = ttng.warp_group_dot
        %next = ttng.warp_group_dot %a, %other, %acc1 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
        // CHECK-NEXT: ttng.warp_group_dot_commit
        // CHECK-NEXT: %[[WAIT_THEN:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]
        // CHECK-SAME: {pendings = 1 : i32}
        // CHECK-NEXT: nvws.consumer_release %{{.*}}, %{{.*}}, %[[SLOT]]
        // CHECK: ttng.warp_group_dot_wait %[[NEXT]]
        nvws.consumer_release %token, %idx, %slot {release_count = 128 : i32} : tensor<2x!nvws.token>, i32, !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
        tt.store %out1, %next : tensor<64x64x!tt.ptr<f32>, #mma>
      } else {
        // CHECK: %[[WAIT_ELSE:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]
        // CHECK-SAME: {pendings = 0 : i32}
        // CHECK-NEXT: nvws.consumer_release %{{.*}}, %{{.*}}, %[[SLOT]]
        nvws.consumer_release %token, %idx, %slot {release_count = 128 : i32} : tensor<2x!nvws.token>, i32, !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      }
      // CHECK: scf.yield %[[DOT]]
      scf.yield %dot, %dot : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    }
    tt.store %out0, %res#0 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.store %out1, %res#1 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @pipe_reader_release_waits_after_branch_local_wgmma
  tt.func @pipe_reader_release_waits_after_branch_local_wgmma(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %slots: !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable>,
      %other: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out0: tensor<64x64x!tt.ptr<f32>, #mma>,
      %out1: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      // CHECK: %[[SLOT:.+]] = ttg.memdesc_index %{{.*}}[%{{.*}}]
      %slot = ttg.memdesc_index %slots[%idx] : !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      // CHECK-NEXT: %[[DOT:.+]] = ttng.warp_group_dot %{{.*}}, %[[SLOT]], %{{.*}}
      %dot = ttng.warp_group_dot %a, %slot, %acc0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[COND:.+]] = arith.cmpi
      %cond = arith.cmpi slt, %iv, %c8 : index
      // CHECK-NEXT: scf.if %[[COND]]
      scf.if %cond {
        // CHECK: %[[NEXT:.+]] = ttng.warp_group_dot
        %next = ttng.warp_group_dot %a, %other, %acc1 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
        // CHECK-NEXT: ttng.warp_group_dot_commit
        // CHECK-NEXT: %[[WAIT_THEN:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]
        // CHECK-SAME: {pendings = 1 : i32}
        // CHECK-NEXT: tle.pipe.reader_release
        // CHECK: ttng.warp_group_dot_wait %[[NEXT]]
        tle.pipe.reader_release %slots[%idx] {capacity = 2 : i32, field_names = ["kv"], pipe_name = "pipe_release_branch", scope = "cta"} : !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable>
        tt.store %out1, %next : tensor<64x64x!tt.ptr<f32>, #mma>
      } else {
        // CHECK: %[[WAIT_ELSE:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]
        // CHECK-SAME: {pendings = 0 : i32}
        // CHECK-NEXT: tle.pipe.reader_release
        tle.pipe.reader_release %slots[%idx] {capacity = 2 : i32, field_names = ["kv"], pipe_name = "pipe_release_branch", scope = "cta"} : !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable>
      }
      // CHECK: scf.yield %[[DOT]]
      scf.yield %dot, %dot : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    }
    tt.store %out0, %res#0 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.store %out1, %res#1 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @release_after_wgmma_c_chain_waits_after_chain
  tt.func @release_after_wgmma_c_chain_waits_after_chain(
      %a0: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b0: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %a1: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b1: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot
      %dot0 = ttng.warp_group_dot %a0, %b0, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot {{.*}}, %[[DOT0]]
      // CHECK-SAME: tle.wgmma_accumulator_chain_c
      %dot1 = ttng.warp_group_dot %a1, %b1, %dot0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT1]]
      // CHECK-SAME: {pendings = 0 : i32}
      // CHECK-NEXT: nvws.consumer_release
      nvws.consumer_release %token, %idx {release_count = 128 : i32} : tensor<2x!nvws.token>, i32
      tt.store %out, %dot1 : tensor<64x64x!tt.ptr<f32>, #mma>
    }
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @release_between_wgmma_c_chain_forces_wait_before_release
  tt.func @release_between_wgmma_c_chain_forces_wait_before_release(
      %a0: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b0: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %a1: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b1: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot
      %dot0 = ttng.warp_group_dot %a0, %b0, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT0:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT0]]
      // CHECK-SAME: {pendings = 0 : i32}
      // CHECK-NEXT: nvws.consumer_release
      nvws.consumer_release %token, %idx {release_count = 128 : i32} : tensor<2x!nvws.token>, i32
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot {{.*}}, %[[WAIT0]]#0
      %dot1 = ttng.warp_group_dot %a1, %b1, %dot0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT1:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT1]]
      // CHECK-SAME: {pendings = 0 : i32}
      // CHECK: tt.store %{{.*}}, %[[WAIT1]]#0
      tt.store %out, %dot1 : tensor<64x64x!tt.ptr<f32>, #mma>
    }
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @release_after_later_wgmma_uses_shared_wait_anchor
  tt.func @release_after_later_wgmma_uses_shared_wait_anchor(
      %a0: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b0: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %a1: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b1: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out0: tensor<64x64x!tt.ptr<f32>, #mma>,
      %out1: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %res:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot
      %dot0 = ttng.warp_group_dot %a0, %b0, %acc0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot
      %dot1 = ttng.warp_group_dot %a1, %b1, %acc1 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT0]], %[[DOT1]]
      // CHECK-SAME: {pendings = 0 : i32}
      // CHECK-NEXT: nvws.consumer_release
      nvws.consumer_release %token, %idx {release_count = 128 : i32} : tensor<2x!nvws.token>, i32
      scf.yield %dot0, %dot1 : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    }
    tt.store %out0, %res#0 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.store %out1, %res#1 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @plain_arrive_barrier_does_not_force_lifetime_drain
  tt.func @plain_arrive_barrier_does_not_force_lifetime_drain(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %barrier: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
      %dot = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: ttng.arrive_barrier {{.*}} {release_fence = true}
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]{{.*}} {pendings = 0 : i32}
      ttng.arrive_barrier %barrier, 128 {release_fence = true} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      scf.yield %dot : tensor<64x64xf32, #mma>
    }
    tt.store %out, %res : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @release_forces_wait_before_lifetime_boundary
  tt.func @release_forces_wait_before_lifetime_boundary(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
      %dot = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]{{.*}} {pendings = 0 : i32}
      // CHECK-NEXT: nvws.consumer_release
      nvws.consumer_release %token, %idx {release_count = 128 : i32} : tensor<2x!nvws.token>, i32
      scf.yield %dot : tensor<64x64xf32, #mma>
    }
    tt.store %out, %res : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @release_old_slot_keeps_later_wgmma_pending
  tt.func @release_old_slot_keeps_later_wgmma_pending(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %slots: !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable>,
      %out0: tensor<64x64x!tt.ptr<f32>, #mma>,
      %out1: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %slot0 = ttg.memdesc_index %slots[%c0_i32] : !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
    %slot1 = ttg.memdesc_index %slots[%c1_i32] : !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %res:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot %{{.*}}, %[[SLOT0:.+]], %{{.*}}
      %dot0 = ttng.warp_group_dot %a, %slot0, %acc0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot %{{.*}}, %[[SLOT1:.+]], %{{.*}}
      %dot1 = ttng.warp_group_dot %a, %slot1, %acc1 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT0]]{{.*}}%[[SLOT0]]
      // CHECK-SAME: {pendings = 1 : i32}
      // CHECK-NEXT: nvws.consumer_release %{{.*}}, %{{.*}}, %[[SLOT0]]
      nvws.consumer_release %token, %idx, %slot0 {release_count = 128 : i32} : tensor<2x!nvws.token>, i32, !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      scf.yield %dot0, %dot1 : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    }
    tt.store %out0, %res#0 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.store %out1, %res#1 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @release_latest_slot_waits_all_pending_wgmma
  tt.func @release_latest_slot_waits_all_pending_wgmma(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %slots: !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable>,
      %out0: tensor<64x64x!tt.ptr<f32>, #mma>,
      %out1: tensor<64x64x!tt.ptr<f32>, #mma>,
      %idx: i32) {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %slot0 = ttg.memdesc_index %slots[%c0_i32] : !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
    %slot1 = ttg.memdesc_index %slots[%c1_i32] : !ttg.memdesc<2x64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
    %slot1_view = tle.memdesc_wgmma_view %slot1 {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared1, #smem>
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    %res:2 = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc0 = %zero, %acc1 = %zero) -> (tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT0:.+]] = ttng.warp_group_dot %{{.*}}, %[[SLOT0:.+]], %{{.*}}
      %dot0 = ttng.warp_group_dot %a, %slot0, %acc0 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[DOT1:.+]] = ttng.warp_group_dot %{{.*}}, %[[SLOT1_VIEW:.+]], %{{.*}}
      %dot1 = ttng.warp_group_dot %a, %slot1_view, %acc1 {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT0]], %[[DOT1]]
      // CHECK-SAME: {pendings = 0 : i32}
      // CHECK-NEXT: nvws.consumer_release %{{.*}}, %{{.*}}, %{{.*}}
      nvws.consumer_release %token, %idx, %slot1 {release_count = 128 : i32} : tensor<2x!nvws.token>, i32, !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>
      scf.yield %dot0, %dot1 : tensor<64x64xf32, #mma>, tensor<64x64xf32, #mma>
    }
    tt.store %out0, %res#0 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.store %out1, %res#1 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#out_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @ready_arrive_waits_before_wgmma_result_store
  tt.func @ready_arrive_waits_before_wgmma_result_store(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable>,
      %ready: !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>,
      %out_smem: !ttg.memdesc<64x64xf32, #out_shared, #smem, mutable>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
      %dot = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem, mutable> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT:.+]]:{{.*}} = ttng.warp_group_dot_wait %[[DOT]]{{.*}} {pendings = 0 : i32}
      // CHECK-NEXT: ttg.local_store %[[WAIT]]#0, %{{.*}}
      // CHECK-NEXT: ttng.arrive_barrier {{.*}} {release_fence = true}
      ttg.local_store %dot, %out_smem : tensor<64x64xf32, #mma> -> !ttg.memdesc<64x64xf32, #out_shared, #smem, mutable>
      ttng.arrive_barrier %ready, 128 {release_fence = true} : !ttg.memdesc<1xi64, #barrier_shared, #smem, mutable>
      scf.yield %dot : tensor<64x64xf32, #mma>
    }
    tt.store %out, %res : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}
