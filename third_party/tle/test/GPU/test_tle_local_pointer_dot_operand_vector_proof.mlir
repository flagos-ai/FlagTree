// RUN: triton-opt %s -split-input-file -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=90 ptx-version=80}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=80})' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The right dot operand has many registers along K, but adjacent addresses
  // within a register group span only two columns. Lane 1 starts at byte 4.
  // CHECK-LABEL: @right_dot_operand_vector_proof
  tt.func public @right_dot_operand_vector_proof() {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %row = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #dot}>>
    %row_e = tt.expand_dims %row {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #dot}>> -> tensor<16x1xi32, #dot>
    %rows = tt.broadcast %row_e : tensor<16x1xi32, #dot> -> tensor<16x32xi32, #dot>
    %col = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot}>>
    %col_e = tt.expand_dims %col {axis = 0 : i32} : tensor<32xi32, #ttg.slice<{dim = 0, parent = #dot}>> -> tensor<1x32xi32, #dot>
    %cols = tt.broadcast %col_e : tensor<1x32xi32, #dot> -> tensor<16x32xi32, #dot>
    %ptr = "tle.local_pointers"(%shared, %rows, %cols) : (!ttg.memdesc<32x32xf16, #shared, #smem, mutable>, tensor<16x32xi32, #dot>, tensor<16x32xi32, #dot>) -> tensor<16x32x!tt.ptr<f16, 3>, #dot>
    // CHECK-NOT: ld.shared.v4.b32
    // CHECK-NOT: ld.shared.v2.b32
    // CHECK: ld.shared.b32
    // CHECK-NOT: ld.shared.v4.b32
    // CHECK-NOT: ld.shared.v2.b32
    %values = tt.load %ptr : tensor<16x32x!tt.ptr<f16, 3>, #dot>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The left operand's register groups run along contiguous K addresses.
  // CHECK-LABEL: @left_dot_operand_vector_proof
  tt.func public @left_dot_operand_vector_proof() {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<32x32xf16, #shared, #smem, mutable>
    %row = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #dot}>>
    %row_e = tt.expand_dims %row {axis = 1 : i32} : tensor<32xi32, #ttg.slice<{dim = 1, parent = #dot}>> -> tensor<32x1xi32, #dot>
    %rows = tt.broadcast %row_e : tensor<32x1xi32, #dot> -> tensor<32x16xi32, #dot>
    %col = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #dot}>>
    %col_e = tt.expand_dims %col {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #dot}>> -> tensor<1x16xi32, #dot>
    %cols = tt.broadcast %col_e : tensor<1x16xi32, #dot> -> tensor<32x16xi32, #dot>
    %ptr = "tle.local_pointers"(%shared, %rows, %cols) : (!ttg.memdesc<32x32xf16, #shared, #smem, mutable>, tensor<32x16xi32, #dot>, tensor<32x16xi32, #dot>) -> tensor<32x16x!tt.ptr<f16, 3>, #dot>
    // CHECK: ld.shared.v4.b32
    %values = tt.load %ptr : tensor<32x16x!tt.ptr<f16, 3>, #dot>
    tt.return
  }
}
