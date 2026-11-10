// RUN: triton-opt %s -split-input-file -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=90 ptx-version=80}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=80})' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Row 1 flips the low physical column bit. Logical columns [0,1,2,3]
  // address physical columns [1,0,3,2], not a contiguous aligned vector.
  // CHECK-LABEL: @swizzled_local_pointer
  tt.func public @swizzled_local_pointer() {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %row = arith.constant dense<1> : tensor<64xi32, #blocked>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%shared, %row, %col) : (!ttg.memdesc<2x64xf32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<f32, 3>, #blocked>
    // CHECK-NOT: ld.shared.v4.b32
    // CHECK: ld.shared.b32
    %value = tt.load %ptr : tensor<64x!tt.ptr<f32, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // A vec=4 swizzle preserves each aligned four-element physical group.
  // CHECK-LABEL: @swizzle_vec4_local_pointer
  tt.func public @swizzle_vec4_local_pointer() {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %row = arith.constant dense<1> : tensor<64xi32, #blocked>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%shared, %row, %col) : (!ttg.memdesc<2x64xf32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<f32, 3>, #blocked>
    // CHECK: ld.shared.v4.b32
    %value = tt.load %ptr : tensor<64x!tt.ptr<f32, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Logical adjacent columns are physically separated by the row extent.
  // CHECK-LABEL: @transposed_local_pointer
  tt.func public @transposed_local_pointer() {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %row = arith.constant dense<1> : tensor<64xi32, #blocked>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%shared, %row, %col) : (!ttg.memdesc<2x64xf32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<f32, 3>, #blocked>
    // CHECK-NOT: ld.shared.v4.b32
    // CHECK: ld.shared.b32
    %value = tt.load %ptr : tensor<64x!tt.ptr<f32, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The scalar base at logical [1,0] is physically element 65, not 64.
  // CHECK-LABEL: @swizzled_scalar_pointer
  tt.func public @swizzled_scalar_pointer() {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %row = arith.constant 1 : i32
    %col = arith.constant 0 : i32
    %base = "tle.local_pointers"(%shared, %row, %col) : (!ttg.memdesc<2x64xf32, #shared, #smem, mutable>, i32, i32) -> !tt.ptr<f32, 3>
    %bases = tt.splat %base : !tt.ptr<f32, 3> -> tensor<32x!tt.ptr<f32, 3>, #blocked>
    %offsets = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    %ptrs = tt.addptr %bases, %offsets : tensor<32x!tt.ptr<f32, 3>, #blocked>, tensor<32xi32, #blocked>
    // CHECK-NOT: ld.shared.v4.b32
    // CHECK: ld.shared.b32
    %value = tt.load %ptrs : tensor<32x!tt.ptr<f32, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.padded_shared<[2:+1] {order = [1, 0], shape = [2, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Padding must preserve both the vector group and its first-address alignment.
  // CHECK-LABEL: @padded_local_pointer
  tt.func public @padded_local_pointer() {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %row = arith.constant dense<1> : tensor<64xi32, #blocked>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%shared, %row, %col) : (!ttg.memdesc<2x64xf32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<f32, 3>, #blocked>
    // CHECK-NOT: ld.shared.v4.b32
    // CHECK: ld.shared.b32
    %value = tt.load %ptr : tensor<64x!tt.ptr<f32, 3>, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.padded_shared<[16:+4] {order = [1, 0], shape = [2, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Padding must preserve both the vector group and its first-address alignment.
  // CHECK-LABEL: @padded_vec4_local_pointer
  tt.func public @padded_vec4_local_pointer() {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %row = arith.constant dense<1> : tensor<64xi32, #blocked>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%shared, %row, %col) : (!ttg.memdesc<2x64xf32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<f32, 3>, #blocked>
    // CHECK: ld.shared.v4.b32
    %value = tt.load %ptr : tensor<64x!tt.ptr<f32, 3>, #blocked>
    tt.return
  }
}
