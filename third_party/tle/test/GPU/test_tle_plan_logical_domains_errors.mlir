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
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: triton-opt %s -triton-tle-plan-logical-domains -verify-diagnostics

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#sharedT = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @plan(%base: !tt.ptr<f16>, %a: !ttg.memdesc<64x256xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %shape0 = arith.constant 0 : i32
    %stride0 = arith.constant 0 : i64
    %desc = tt.make_tensor_descriptor %base, [%shape0, %shape0], [%stride0, %stride0] {tle.logical_descriptor_shape = array<i64: 80, 256>} : <f16>, <tensor<128x256xf16, #shared>>
    // expected-note @+1 {{logical domain root is here}}
    %root = ttg.local_alloc {tle.logical_alloc_shape = array<i64: 1, 80, 256>, tle.logical_non_power_axis = 1 : i32, tle.storage_plan = "candidate"} : () -> !ttg.memdesc<1x128x256xf16, #shared, #smem, mutable>
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %stage = ttg.memdesc_index %root[%c0] : !ttg.memdesc<1x128x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x256xf16, #shared, #smem, mutable>
    // expected-error @+1 {{expect_bytes must be a multiple of the logical TMA byte count (40960)}}
    ttg.tma_copy %desc, %stage, [%c0, %c0], barrier %bar {expect_bytes = 40961 : i32, tle.logical_copy_shape = array<i64: 80, 256>} : !tt.tensordesc<tensor<128x256xf16, #shared>>, !ttg.memdesc<128x256xf16, #shared, #smem, mutable>, !ttg.memdesc<1xi64, #bar, #smem, mutable>
    %stage_t = ttg.memdesc_trans %stage {order = array<i32: 1, 0>} : !ttg.memdesc<128x256xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x128xf16, #sharedT, #smem, mutable>
    %zero = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #mma>
    %dot = tle.wgmma %a, %stage_t, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x256xf16, #shared, #smem, mutable> * !ttg.memdesc<256x128xf16, #sharedT, #smem, mutable>, tensor<64x128xf32, #mma> -> tensor<64x128xf32, #mma>
    %wait = tle.wgmma_wait %dot {pendings = 0 : i32} : tensor<64x128xf32, #mma> -> tensor<64x128xf32, #mma>
    tt.return
  }
}
