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

// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1250 --convert-builtin-func-to-llvm | FileCheck %s --check-prefix=GFX1250

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx1250", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // GFX1250-LABEL: init_barrier
  tt.func @init_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // GFX1250: %[[INIT_VAL1:.+]] = llvm.mlir.constant(4294967297 : i64) : i64
    // GFX1250: %[[ALLOC_PTR:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<3>, i32)>
    // GFX1250: llvm.store %[[INIT_VAL1]], %[[ALLOC_PTR]] : i64, !llvm.ptr<3>
    // GFX1250: rocdl.barrier
    amdg.init_barrier %alloc, 2 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // GFX1250-LABEL: wait_barrier
  tt.func @wait_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>, %phase: i32) {
    // GFX1250: rocdl.s.sleep {{.*}}
    // GFX1250: llvm.load {{.*}} : !llvm.ptr<3> -> i64
    // GFX1250: llvm.icmp "ne" {{%arg1, %.*|%.*, %arg1}} : i32
    amdg.wait_barrier %alloc, %phase : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }

  // GFX1250-LABEL: arrive_barrier
  tt.func @arrive_barrier(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // GFX1250: %[[UPDATE_VAL1:.+]] = llvm.mlir.constant(1 : i64) : i64
    // GFX1250: %[[ALLOC_PTR:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<3>, i32)>
    // GFX1250: llvm.call_intrinsic "llvm.amdgcn.ds.atomic.barrier.arrive.rtn.b64"(%[[ALLOC_PTR]], %[[UPDATE_VAL1]])
    %prior_phase = amdg.arrive_barrier %alloc, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable> -> i32
    tt.return
  }

  // GFX1250-LABEL: async_copy_mbarrier_arrive
  tt.func @async_copy_mbarrier_arrive(%alloc: !ttg.memdesc<1xi64, #shared, #smem, mutable>) {
    // GFX1250: %[[ALLOC_PTR:.+]] = llvm.extractvalue %arg0[0] : !llvm.struct<(ptr<3>, i32)>
    // GFX1250: llvm.call_intrinsic "llvm.amdgcn.ds.atomic.async.barrier.arrive.b64"(%[[ALLOC_PTR]])
    amdg.async_copy_mbarrier_arrive %alloc : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
