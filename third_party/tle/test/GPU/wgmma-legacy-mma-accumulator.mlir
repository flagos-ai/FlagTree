// Copyright 2026 FlagOS Contributors
// RUN: triton-opt %s -triton-tle-lower-wgmma | FileCheck %s

// A ready MMA2 accumulator is ordinary input to WGMMA. Convert it to Hopper
// MMA before issuing the async operation, and restore its consumer layout
// only after the wait. MMA2 does not identify an already-lowered async value.
// A terminal 64x64 WGMMA with eight warps uses the shared MMAv3 [4, 2] split.
#legacy = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
// CHECK-DAG: #[[$LEGACY:.*]] = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 8], instrShape = [16, 8]}>
// CHECK-DAG: #[[$HOPPER:.*]] = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 32, 16]}>

module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: tt.func @pv_from_legacy_accumulator
  tt.func @pv_from_legacy_accumulator(%a: tensor<64x64xbf16, #legacy>, %b: !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>, %c: tensor<64x64xf32, #legacy>) -> tensor<64x64xf32, #legacy> {
    // CHECK: %[[C:.*]] = ttg.convert_layout %arg2 : tensor<64x64xf32, #[[$LEGACY]]> -> tensor<64x64xf32, #[[$HOPPER]]>
    // CHECK: %[[A:.*]] = ttg.convert_layout %arg0
    // CHECK: %[[DOT:.*]] = ttng.warp_group_dot %[[A]], %arg1, %[[C]] {{.*}} -> tensor<64x64xf32, #[[$HOPPER]]>
    %pending = tle.wgmma %a, %b, %c {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x64xbf16, #legacy> * !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>, tensor<64x64xf32, #legacy> -> tensor<64x64xf32, #legacy>
    // CHECK: %[[WAIT:.*]] = ttng.warp_group_dot_wait %[[DOT]]
    %ready = tle.wgmma_wait %pending {pendings = 0 : i32} : tensor<64x64xf32, #legacy> -> tensor<64x64xf32, #legacy>
    // CHECK: %[[READY:.*]] = ttg.convert_layout %[[WAIT]] : tensor<64x64xf32, #[[$HOPPER]]> -> tensor<64x64xf32, #[[$LEGACY]]>
    // CHECK: tt.return %[[READY]]
    tt.return %ready : tensor<64x64xf32, #legacy>
  }
}
