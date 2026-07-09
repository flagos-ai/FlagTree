// RUN: triton-opt %s -split-input-file -tritongpu-pipeline -canonicalize | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "tle.wgmma_pipeline_mode" = "user_promise"} {
  // CHECK-LABEL: tt.func @explicit_wait_is_preserved_without_extra_drain
  tt.func @explicit_wait_is_preserved_without_extra_drain(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // CHECK: %[[RES:.+]] = scf.for
    %res = scf.for %iv = %c0 to %c8 step %c1 iter_args(%acc = %zero) -> (tensor<64x64xf32, #mma>) {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
      // CHECK-SAME: {inputPrecision = 0 : i32, isAsync = true, tle.explicit_wgmma_commit}
      %dot = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NEXT: %[[WAIT1:.+]] = ttng.warp_group_dot_wait %[[DOT]]
      // CHECK-SAME: {pendings = 1 : i32}
      %wait1 = ttng.warp_group_dot_wait %dot {pendings = 1 : i32} : tensor<64x64xf32, #mma>
      // CHECK-NOT: ttng.warp_group_dot_wait
      // CHECK: scf.yield %[[WAIT1]]
      scf.yield %wait1 : tensor<64x64xf32, #mma>
    }
    // CHECK: %[[WAIT0:.+]] = ttng.warp_group_dot_wait %[[RES]]
    // CHECK-SAME: {pendings = 0 : i32}
    %wait0 = ttng.warp_group_dot_wait %res {pendings = 0 : i32} : tensor<64x64xf32, #mma>
    // CHECK: tt.store %{{.*}}, %[[WAIT0]]
    tt.store %out, %wait0 : tensor<64x64x!tt.ptr<f32>, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, "tle.wgmma_pipeline_mode" = "user_promise"} {
  // CHECK-LABEL: tt.func @ordinary_use_gets_no_auto_wait
  tt.func @ordinary_use_gets_no_auto_wait(
      %a: !ttg.memdesc<64x64xbf16, #shared, #smem>,
      %b: !ttg.memdesc<64x64xbf16, #shared1, #smem>,
      %out: tensor<64x64x!tt.ptr<f32>, #mma>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    scf.for %iv = %c0 to %c8 step %c1 {
      // CHECK: %[[DOT:.+]] = ttng.warp_group_dot
      // CHECK-SAME: {inputPrecision = 0 : i32, isAsync = true, tle.explicit_wgmma_commit}
      %dot = ttng.warp_group_dot %a, %b, %zero {inputPrecision = 0 : i32} : !ttg.memdesc<64x64xbf16, #shared, #smem> * !ttg.memdesc<64x64xbf16, #shared1, #smem> -> tensor<64x64xf32, #mma>
      // CHECK-NEXT: ttng.warp_group_dot_commit
      // CHECK-NOT: ttng.warp_group_dot_wait
      // CHECK: tt.store %{{.*}}, %[[DOT]]
      tt.store %out, %dot : tensor<64x64x!tt.ptr<f32>, #mma>
    }
    tt.return
  }
}
