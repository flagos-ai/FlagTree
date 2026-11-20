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

// -----

// An explicit layout on a vector expanded inside a loop stays local: the
// synthesized expand_dims parent the RLC phases derive from it is only a
// candidate, so the loop accumulator keeps its layout.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @explicit_vector_in_loop
  // CHECK: scf.for {{.*}} -> (tensor<16x512xf32, #blocked>)
  // CHECK: tt.expand_dims %{{.*}} {axis = 0 : i32} : tensor<512xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x512xf32, #blocked>
  // CHECK-NOT: ttg.convert_layout
  // CHECK: tt.return
  tt.func public @explicit_vector_in_loop(%v: tensor<512xf32, #ttg.slice<{dim = 0, parent = #blocked}>>, %n: i32) -> tensor<16x512xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %zero = arith.constant dense<0.000000e+00> : tensor<16x512xf32, #blocked>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %zero) -> (tensor<16x512xf32, #blocked>) : i32 {
      %h = ttg.convert_layout %v {tle.explicit_encoding.0 = #blocked1} : tensor<512xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<512xf32, #blocked1>
      %e0 = math.exp %h : tensor<512xf32, #blocked1>
      %s = ttg.convert_layout %e0 : tensor<512xf32, #blocked1> -> tensor<512xf32, #ttg.slice<{dim = 0, parent = #blocked}>>
      %x = tt.expand_dims %s {axis = 0 : i32} : tensor<512xf32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x512xf32, #blocked>
      %b = tt.broadcast %x : tensor<1x512xf32, #blocked> -> tensor<16x512xf32, #blocked>
      %next = arith.addf %acc, %b : tensor<16x512xf32, #blocked>
      scf.yield %next : tensor<16x512xf32, #blocked>
    }
    tt.return %r : tensor<16x512xf32, #blocked>
  }
}
