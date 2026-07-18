// RUN: triton-opt %s -tritongpu-pipeline | FileCheck %s --implicit-check-not=ttg.mask

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @dynamic_predicated_wgmma_stage
  // CHECK: %[[PROLOGUE_PRED:.+]] = arith.cmpi sgt
  // CHECK: scf.if %[[PROLOGUE_PRED]] -> (tensor<64x64xf32, #mma>)
  // CHECK:   ttng.warp_group_dot
  // CHECK:   scf.yield %{{.*}} : tensor<64x64xf32, #mma>
  // CHECK: } else {
  // CHECK:   scf.yield %arg2 : tensor<64x64xf32, #mma>
  // CHECK: scf.for
  // CHECK:   %[[KERNEL_PRED:.+]] = arith.cmpi slt
  // CHECK:   scf.if %[[KERNEL_PRED]] -> (tensor<64x64xf32, #mma>)
  // CHECK:     ttng.warp_group_dot {{.*}} {inputPrecision = 0 : i32, isAsync = true, tle.explicit_wgmma_commit}
  // CHECK-NEXT: ttng.warp_group_dot_commit
  // CHECK-NEXT: %{{.*}}:3 = ttng.warp_group_dot_wait
  // CHECK:   } else {
  // CHECK:     scf.yield %{{.*}} : tensor<64x64xf32, #mma>
  tt.func @dynamic_predicated_wgmma_stage(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem>,
      %acc: tensor<64x64xf32, #mma>,
      %ub: i32) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %r = scf.for %iv = %c0 to %ub step %c1
        iter_args(%arg = %acc) -> tensor<64x64xf32, #mma> : i32 {
      %dot = ttng.warp_group_dot %a, %b, %arg
          {inputPrecision = 0 : i32, loop.cluster = 0 : i32, loop.stage = 0 : i32}
          : !ttg.memdesc<64x64xbf16, #shared, #smem> *
            !ttg.memdesc<64x64xbf16, #shared1, #smem>
            -> tensor<64x64xf32, #mma>
      %unused = arith.addi %iv, %c0
          {loop.cluster = 0 : i32, loop.stage = 3 : i32} : i32
      scf.yield %dot : tensor<64x64xf32, #mma>
    } {tt.num_stages = 4 : i32, tt.scheduled_max_stage = 3 : i32}
    tt.return %r : tensor<64x64xf32, #mma>
  }
}
