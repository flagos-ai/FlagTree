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

// RUN: triton-opt %s --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts 2>/dev/null | FileCheck %s --dump-input-context 20

#shared3 = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [2, 1, 0]}>
#shared2 = #ttg.swizzled_shared<{vec = 2, perPhase = 2, maxPhase = 4, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: rank_reducing_subview_of_shared_allocation
  tt.func @rank_reducing_subview_of_shared_allocation() {
    // The logical field contains 32x64 elements, but one backing stage is
    // 64x64 elements. Indexing stage 1 must therefore use a stride of 4096.
    // CHECK: llvm.mlir.constant(4096 : i32) : i32
    // CHECK: llvm.mul
    // CHECK: llvm.getelementptr
    %c1 = arith.constant 1 : i32
    %storage = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xf32, #shared3, #smem, mutable>
    %field = ttg.memdesc_subslice %storage[0, 32, 0] : !ttg.memdesc<2x64x64xf32, #shared3, #smem, mutable> -> !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64>
    %slot = ttg.memdesc_index %field[%c1] : !ttg.memdesc<2x32x64xf32, #shared3, #smem, mutable, 2x64x64> -> !ttg.memdesc<32x64xf32, #shared2, #smem, mutable, 2x64x64>
    tt.return
  }
}
