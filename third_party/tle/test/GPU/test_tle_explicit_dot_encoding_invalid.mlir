// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' -verify-diagnostics

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @unsupported_accumulator_layout() {
    %a = arith.constant dense<1.0> : tensor<64x32xf16>
    %b = arith.constant dense<1.0> : tensor<32x64xf16>
    %c = arith.constant dense<0.0> : tensor<64x64xf32>
    %acc = tle.gpu.set_layout %c {target_encoding = #mma} : tensor<64x64xf32> -> tensor<64x64xf32>
    // expected-error @+1 {{explicit MMA dot layout currently requires NVIDIA MMA v2}}
    %d = tt.dot %a, %b, %acc : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @unsupported_result_layout() {
    %a = arith.constant dense<1.0> : tensor<64x32xf16>
    %b = arith.constant dense<1.0> : tensor<32x64xf16>
    %c = arith.constant dense<0.0> : tensor<64x64xf32>
    // expected-error @+1 {{explicit MMA dot layout currently requires NVIDIA MMA v2}}
    %d = tt.dot %a, %b, %c : tensor<64x32xf16> * tensor<32x64xf16> -> tensor<64x64xf32>
    %fixed = tle.gpu.set_layout %d {target_encoding = #mma} : tensor<64x64xf32> -> tensor<64x64xf32>
    tt.return
  }
}
