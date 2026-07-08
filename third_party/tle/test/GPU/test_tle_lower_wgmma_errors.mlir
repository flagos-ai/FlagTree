// RUN: triton-opt %s -split-input-file -triton-tle-lower-wgmma -verify-diagnostics

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @direct_use_without_wait(
      %a: !ttg.memdesc<64x16xf16, #shared, #smem, mutable>,
      %b: !ttg.memdesc<16x16xf16, #shared, #smem, mutable>,
      %out: tensor<64x16x!tt.ptr<f32>, #blocked>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    // expected-error @+1 {{async result must be consumed by tle.wgmma_wait before ordinary tensor use}}
    %dot = tle.wgmma %a, %b, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> * !ttg.memdesc<16x16xf16, #shared, #smem, mutable>, tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    tt.store %out, %dot : tensor<64x16x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wait_without_wgmma(%acc: tensor<64x16xf32, #blocked>) {
    // expected-error @+1 {{input must be the async result of tle.wgmma}}
    %wait = tle.wgmma_wait %acc {pendings = 0 : i32} : tensor<64x16xf32, #blocked> -> tensor<64x16xf32, #blocked>
    tt.return
  }
}
