// Copyright 2026 FlagOS Contributors
// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' --convert-nv-gpu-to-llvm | FileCheck %s

// Complete descriptor operands must remain SSA values: block-local PTX
// temporaries can be initialized only in the peeled iteration after ptxas
// unrolling and read uninitialized on the runtime loop's other paths.
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 32]}>
#fp8 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 8}>
#fp8t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 8}>
#fp16 = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#fp16t = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @runtime_loop_descriptors
  // CHECK: llvm.add
  // CHECK-NOT: __tle_wgmma_desc
  // CHECK: wgmma.mma_async.sync.aligned.m64n64k32
  // CHECK-NOT: __tle_wgmma_desc
  // CHECK: wgmma.mma_async.sync.aligned.m64n64k16
  // CHECK-NOT: __tle_wgmma_desc
  // CHECK: llvm.return
  tt.func @runtime_loop_descriptors(%a0: !ttg.memdesc<64x32xf8E4M3FN, #fp8, #smem>, %b0: !ttg.memdesc<32x64xf8E4M3FN, #fp8t, #smem>, %a1: !ttg.memdesc<64x16xf16, #fp16, #smem>, %b1: !ttg.memdesc<16x64xf16, #fp16t, #smem>, %initial: tensor<64x64xf32, #mma>, %inner: i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %result = scf.for %i = %zero to %inner step %one iter_args(%acc = %initial) -> tensor<64x64xf32, #mma> : i32 {
      %first = ttng.warp_group_dot %a0, %b0, %acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 1073741824 : i32} : !ttg.memdesc<64x32xf8E4M3FN, #fp8, #smem> * !ttg.memdesc<32x64xf8E4M3FN, #fp8t, #smem> -> tensor<64x64xf32, #mma>
      %second = ttng.warp_group_dot %a1, %b1, %first {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<64x16xf16, #fp16, #smem> * !ttg.memdesc<16x64xf16, #fp16t, #smem> -> tensor<64x64xf32, #mma>
      scf.yield %second : tensor<64x64xf32, #mma>
    }
    %ready = ttng.warp_group_dot_wait %result {pendings = 0 : i32} : tensor<64x64xf32, #mma>
    tt.return
  }
}
