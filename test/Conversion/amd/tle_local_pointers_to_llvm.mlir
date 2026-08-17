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

// RUN: triton-opt %s -split-input-file --allocate-amdgpu-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx1201 --convert-builtin-func-to-llvm | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @local_pointers_tensor_index
  // The lowered pointers must stay in the shared address space and be reached
  // through plain GEPs, with no NVVM op leaking into the AMD path.
  // CHECK: llvm.mlir.addressof @global_smem : !llvm.ptr<3>
  // CHECK: llvm.getelementptr {{.*}}!llvm.ptr<3>
  // CHECK: llvm.load {{.*}}!llvm.ptr<3>
  // CHECK: llvm.store {{.*}}!llvm.ptr<3>
  // CHECK-NOT: tle.local_pointers
  // CHECK-NOT: nvvm.
  tt.func public @local_pointers_tensor_index(%idx: tensor<32xi32, #blocked>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %ptrs = "tle.local_pointers"(%buf, %idx) : (!ttg.memdesc<32xf32, #shared, #smem, mutable>, tensor<32xi32, #blocked>) -> tensor<32x!tt.ptr<f32, 3>, #blocked>
    %val = tt.load %ptrs : tensor<32x!tt.ptr<f32, 3>, #blocked>
    tt.store %ptrs, %val : tensor<32x!tt.ptr<f32, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // A scalar base pointer broadcast over a tensor of offsets must not be
  // rewritten into buffer ops, which can only address global memory.
  // CHECK-LABEL: llvm.func @local_pointers_scalar_index
  // CHECK: llvm.getelementptr {{.*}}!llvm.ptr<3>
  // CHECK-NOT: tle.local_pointers
  // CHECK-NOT: amdg.buffer_
  tt.func public @local_pointers_scalar_index(%idx: i32) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%buf, %idx) : (!ttg.memdesc<32xf32, #shared, #smem, mutable>, i32) -> !tt.ptr<f32, 3>
    %splat = tt.splat %ptr : !tt.ptr<f32, 3> -> tensor<32x!tt.ptr<f32, 3>, #blocked>
    %offs = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %ptrs = tt.addptr %splat, %offs : tensor<32x!tt.ptr<f32, 3>, #blocked>, tensor<32xi32, #blocked>
    %val = tt.load %ptrs : tensor<32x!tt.ptr<f32, 3>, #blocked>
    tt.store %ptrs, %val : tensor<32x!tt.ptr<f32, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1201", "ttg.threads-per-warp" = 32 : i32} {
  // tle.local_pointers has no axis-info visitor, so its result carries a
  // degenerate rank-0 entry. The atomic lowering must tolerate that instead of
  // indexing the contiguity vector, which is only a hint for intra-wave reduce.
  // CHECK-LABEL: llvm.func @local_pointers_atomic_rmw
  // CHECK: llvm.getelementptr {{.*}}!llvm.ptr<3>
  // CHECK: llvm.atomicrmw fadd {{.*}} syncscope("workgroup") monotonic : !llvm.ptr<3>, f32
  // CHECK-NOT: tle.local_pointers
  tt.func public @local_pointers_atomic_rmw(%idx: tensor<32xi32, #blocked>, %val: tensor<32xf32, #blocked>) {
    %buf = ttg.local_alloc : () -> !ttg.memdesc<32xf32, #shared, #smem, mutable>
    %ptrs = "tle.local_pointers"(%buf, %idx) : (!ttg.memdesc<32xf32, #shared, #smem, mutable>, tensor<32xi32, #blocked>) -> tensor<32x!tt.ptr<f32, 3>, #blocked>
    %res = tt.atomic_rmw fadd, relaxed, cta, %ptrs, %val : (tensor<32x!tt.ptr<f32, 3>, #blocked>, tensor<32xf32, #blocked>) -> tensor<32xf32, #blocked>
    tt.return
  }
}
