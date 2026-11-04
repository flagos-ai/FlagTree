// Copyright 2026 FlagOS Contributors
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=8 enable-source-remat=false' | FileCheck %s

// WGMMA C/D and wait input/output have tied types. Explicit hints must cross
// these edges without constraining a register A operand to the C/D layout.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 8, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 8}>
#smem = #ttg.shared_memory
// CHECK: #[[$MMA:.*]] = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 8, 32]}>

module {
  // CHECK-LABEL: tt.func @hinted_accumulator
  tt.func @hinted_accumulator(%a: !ttg.memdesc<64x128xf8E4M3FN, #shared, #smem, mutable>, %b: !ttg.memdesc<128x8xf8E4M3FN, #shared, #smem, mutable>) {
    %zero = arith.constant dense<0.0> : tensor<64x8xf32>
    %hint = tle.gpu.set_layout %zero {target_encoding = #mma} : tensor<64x8xf32> -> tensor<64x8xf32>
    // CHECK: %[[DOT:.*]] = tle.wgmma {{.*}} tensor<64x8xf32, #[[$MMA]]> -> tensor<64x8xf32, #[[$MMA]]>
    %dot = tle.wgmma %a, %b, %hint {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x128xf8E4M3FN, #shared, #smem, mutable> * !ttg.memdesc<128x8xf8E4M3FN, #shared, #smem, mutable>, tensor<64x8xf32> -> tensor<64x8xf32>
    // CHECK: tle.wgmma_wait %[[DOT]] {{.*}} tensor<64x8xf32, #[[$MMA]]> -> tensor<64x8xf32, #[[$MMA]]>
    %wait = tle.wgmma_wait %dot {pendings = 0 : i32} : tensor<64x8xf32> -> tensor<64x8xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func @hinted_wait_result
  tt.func @hinted_wait_result(%a: !ttg.memdesc<64x128xf8E4M3FN, #shared, #smem, mutable>, %b: !ttg.memdesc<128x8xf8E4M3FN, #shared, #smem, mutable>) {
    %zero = arith.constant dense<0.0> : tensor<64x8xf32>
    // CHECK: %[[DOT:.*]] = tle.wgmma {{.*}} tensor<64x8xf32, #[[$MMA]]> -> tensor<64x8xf32, #[[$MMA]]>
    %dot = tle.wgmma %a, %b, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x128xf8E4M3FN, #shared, #smem, mutable> * !ttg.memdesc<128x8xf8E4M3FN, #shared, #smem, mutable>, tensor<64x8xf32> -> tensor<64x8xf32>
    // CHECK: tle.wgmma_wait %[[DOT]] {{.*}} tensor<64x8xf32, #[[$MMA]]> -> tensor<64x8xf32, #[[$MMA]]>
    %wait = tle.wgmma_wait %dot {pendings = 0 : i32} : tensor<64x8xf32> -> tensor<64x8xf32>
    %hint = tle.gpu.set_layout %wait {target_encoding = #mma} : tensor<64x8xf32> -> tensor<64x8xf32>
    tt.return
  }

  // CHECK-LABEL: tt.func @independent_register_a
  tt.func @independent_register_a(%a: !tt.ptr<f8E4M3FN>, %b: !ttg.memdesc<128x8xf8E4M3FN, #shared, #smem, mutable>) {
    %ap = tt.splat %a : !tt.ptr<f8E4M3FN> -> tensor<64x128x!tt.ptr<f8E4M3FN>>
    %av = tt.load %ap : tensor<64x128x!tt.ptr<f8E4M3FN>>
    %zero = arith.constant dense<0.0> : tensor<64x8xf32>
    %hint = tle.gpu.set_layout %zero {target_encoding = #mma} : tensor<64x8xf32> -> tensor<64x8xf32>
    // CHECK: tle.wgmma %{{.*}}, %{{.*}}, %{{.*}} {{.*}} tensor<64x128xf8E4M3FN, #{{blocked.*}}> * {{.*}}, tensor<64x8xf32, #[[$MMA]]> -> tensor<64x8xf32, #[[$MMA]]>
    %dot = tle.wgmma %av, %b, %hint {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf8E4M3FN> * !ttg.memdesc<128x8xf8E4M3FN, #shared, #smem, mutable>, tensor<64x8xf32> -> tensor<64x8xf32>
    %wait = tle.wgmma_wait %dot {pendings = 0 : i32} : tensor<64x8xf32> -> tensor<64x8xf32>
    tt.return
  }
}
