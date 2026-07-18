// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// RUN: triton-opt %s --split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_load
  tt.func public @tdm_load(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %c_pred = arith.constant true
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <tensor<64x64xf16, #shared>>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK-COUNT-4: llvm.insertelement{{.*}} : vector<4xi32>
    // CHECK-COUNT-8: llvm.insertelement{{.*}} : vector<8xi32>
    // CHECK: llvm.amdgcn.tensor.load.to.lds.d2{{.*}} : (vector<4xi32>, vector<8xi32>, i32) -> ()
    %2 = amdg.async_tdm_copy_global_to_local %0[%c_offset, %c_offset] into %1, %c_pred : !tt.tensordesc<tensor<64x64xf16, #shared>> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK: llvm.amdgcn.s.wait.tensorcnt{{.*}} : (i16) -> ()
    %3 = amdg.async_tdm_wait  {num = 0 : i32}
    %4 = ttg.local_load %1 : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> tensor<64x64xf16, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tdm_store
  tt.func public @tdm_store(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c_shape = arith.constant 128 : i32
    %c_stride0 = arith.constant 128 : i64
    %c_stride1 = arith.constant 1 : i64
    %c_offset = arith.constant 0 : i32
    %0 = tt.make_tensor_descriptor %arg0, [%c_shape, %c_shape], [%c_stride0, %c_stride1] : <f16>, <tensor<64x64xf16, #shared>>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    %2 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked>
    ttg.local_store %2, %1 : tensor<64x64xf16, #blocked> -> !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    // CHECK-COUNT-4: llvm.insertelement{{.*}} : vector<4xi32>
    // CHECK-COUNT-8: llvm.insertelement{{.*}} : vector<8xi32>
    // CHECK: llvm.amdgcn.tensor.store.from.lds.d2{{.*}} : (vector<4xi32>, vector<8xi32>, i32) -> ()
    amdg.async_tdm_copy_local_to_global %0[%c_offset, %c_offset] from %1: !ttg.memdesc<64x64xf16, #shared, #smem, mutable> -> !tt.tensordesc<tensor<64x64xf16, #shared>>
    // CHECK: llvm.amdgcn.s.wait.tensorcnt{{.*}} : (i16) -> ()
    %3 = amdg.async_tdm_wait  {num = 0 : i32}
    tt.return
  }
}
