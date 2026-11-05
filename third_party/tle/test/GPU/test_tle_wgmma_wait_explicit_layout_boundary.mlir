// Copyright 2026 FlagOS Contributors
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8 enable-source-remat=false' | FileCheck %s

// Once the wait has made its result ready, a set_layout use may convert it
// independently. It must not rewrite the async producer's accumulator layout.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 8, 32]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#smem = #ttg.shared_memory
// CHECK: #[[$BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
// CHECK: #[[$MMA:.*]] = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 8, 32]}>
module {
  // CHECK-LABEL: tt.func @ready_result_layout_boundary
  tt.func @ready_result_layout_boundary(%a: !ttg.memdesc<64x128xf8E4M3FN, #shared, #smem, mutable>, %b: !ttg.memdesc<128x8xf8E4M3FN, #shared, #smem, mutable>, %output: !tt.ptr<f32>) {
    %zero = arith.constant dense<0.0> : tensor<64x8xf32>
    %hint = tle.gpu.set_layout %zero {target_encoding = #mma} : tensor<64x8xf32> -> tensor<64x8xf32>
    %dot = tle.wgmma %a, %b, %hint {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x128xf8E4M3FN, #shared, #smem, mutable> * !ttg.memdesc<128x8xf8E4M3FN, #shared, #smem, mutable>, tensor<64x8xf32> -> tensor<64x8xf32>
    // CHECK: %[[WAIT:.*]] = tle.wgmma_wait {{.*}} tensor<64x8xf32, #[[$MMA]]> -> tensor<64x8xf32, #[[$MMA]]>
    %wait = tle.wgmma_wait %dot {pendings = 0 : i32} : tensor<64x8xf32> -> tensor<64x8xf32>
    // CHECK: %[[READY:.*]] = ttg.convert_layout %[[WAIT]] {{.*}} : tensor<64x8xf32, #[[$MMA]]> -> tensor<64x8xf32, #[[$BLOCKED]]>
    %out = tle.gpu.set_layout %wait {target_encoding = #blocked} : tensor<64x8xf32> -> tensor<64x8xf32>
    %pointers = tt.splat %output : !tt.ptr<f32> -> tensor<64x8x!tt.ptr<f32>>
    // CHECK: tt.store {{.*}}, %[[READY]]
    tt.store %pointers, %out : tensor<64x8x!tt.ptr<f32>>
    tt.return
  }
}
