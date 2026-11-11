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

// RUN: triton-opt %s -split-input-file -tritongpu-reorder-instructions=keep-smem-read-order=true | FileCheck %s --check-prefixes=COMMON,ON
// RUN: triton-opt %s -split-input-file -tritongpu-reorder-instructions | FileCheck %s --check-prefixes=COMMON,OFF

// A store through pointers of tle.local_pointers writes shared memory; which
// buffer is not tracked through pointer arithmetic, so the load stays even
// though the pointers are of another allocation.
// COMMON-LABEL: no_sink_load_past_local_pointer_store
//       ON: ttg.local_load
//       ON: scf.for
//       ON:   tt.store
//       ON:   ttg.local_load
//   ON-NOT:   ttg.local_load
//       ON:   tt.dot
//  OFF-NOT: ttg.local_load
//      OFF: scf.for
//      OFF:   tt.store
//      OFF:   ttg.local_load
//      OFF:   ttg.local_load
//      OFF:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_sink_load_past_local_pointer_store(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %S = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %T = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %TP = "tle.local_pointers"(%T) : (!ttg.memdesc<32x32xf32, #shared, #smem, mutable>) -> tensor<32x32x!tt.ptr<f32, 3>, #blocked>
    %AD = ttg.local_load %S : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      tt.store %TP, %B : tensor<32x32x!tt.ptr<f32, 3>, #blocked>
      %BD = ttg.local_load %T : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %d = tt.dot %AD, %BD, %acc, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    %o = ttg.convert_layout %r : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %o : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
