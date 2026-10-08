// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s
// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions="enable-rlc-enhance=true" | FileCheck %s

// The result of a tle.gpu.set_layout convert carries a hard encoding. When it
// meets an MMA value, conflict resolution must keep the explicit layout instead
// of the default MMA preference, with or without the FlagTree RLC phases.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: #[[$L:blocked[0-9]*]] = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
  // CHECK-LABEL: tt.func @explicit_layout_wins
  // CHECK: ttg.convert_layout %{{.*}} {tle.explicit_encoding.0 = #[[$L]]} : tensor<64x64xf32, #{{.*}}> -> tensor<64x64xf32, #[[$L]]>
  // CHECK: arith.addf %{{.*}}, %{{.*}} : tensor<64x64xf32, #[[$L]]>
  // CHECK: math.exp %{{.*}} : tensor<64x64xf32, #[[$L]]>
  tt.func @explicit_layout_wins(%acc: tensor<64x64xf32, #mma>, %x: tensor<64x64xf32, #blocked1>) -> tensor<64x64xf32, #blocked> {
    %a = ttg.convert_layout %acc : tensor<64x64xf32, #mma> -> tensor<64x64xf32, #blocked>
    %c = ttg.convert_layout %x {tle.explicit_encoding.0 = #blocked} : tensor<64x64xf32, #blocked1> -> tensor<64x64xf32, #blocked>
    %z = arith.addf %a, %c : tensor<64x64xf32, #blocked>
    %e = math.exp %z : tensor<64x64xf32, #blocked>
    tt.return %e : tensor<64x64xf32, #blocked>
  }
}
