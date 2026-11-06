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

// RUN: triton-opt %s -split-input-file --allocate-shared-memory -test-print-membar | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @whole_group_wait
  tt.func @whole_group_wait(%values: tensor<128xf32, #blocked>) {
    %id = arith.constant 3 : i32
    %threads = arith.constant 128 : i32
    %buf = ttg.local_alloc %values : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
    // CHECK: ttng.wait_barrier_named
    // CHECK-NEXT: {{.*}}ttg.local_load
    ttng.wait_barrier_named %id, %threads : i32, i32
    %data = ttg.local_load %buf : !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @partial_group_wait
  tt.func @partial_group_wait(%values: tensor<128xf32, #blocked>) {
    %id = arith.constant 3 : i32
    %threads = arith.constant 64 : i32
    %buf = ttg.local_alloc %values : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
    // CHECK: ttng.wait_barrier_named
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: {{.*}}ttg.local_load
    ttng.wait_barrier_named %id, %threads : i32, i32
    %data = ttg.local_load %buf : !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @arrive_does_not_wait
  tt.func @arrive_does_not_wait(%values: tensor<128xf32, #blocked>) {
    %id = arith.constant 3 : i32
    %threads = arith.constant 128 : i32
    %buf = ttg.local_alloc %values : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
    // CHECK: ttng.arrive_barrier_named
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: {{.*}}ttg.local_load
    ttng.arrive_barrier_named %id, %threads : i32, i32
    %data = ttg.local_load %buf : !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @other_target_unchanged
  tt.func @other_target_unchanged(%values: tensor<128xf32, #blocked>) {
    %id = arith.constant 3 : i32
    %threads = arith.constant 128 : i32
    %buf = ttg.local_alloc %values : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
    // CHECK: ttng.wait_barrier_named
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: {{.*}}ttg.local_load
    ttng.wait_barrier_named %id, %threads : i32, i32
    %data = ttg.local_load %buf : !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

// A pipe completion wait does not itself synchronize readers. Keep the write
// dependency so the first actual local load still gets a group barrier.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipe_wait_keeps_dependency
  tt.func @pipe_wait_keeps_dependency(%values: tensor<128xf32, #blocked>, %a: i32) {
    %buf = ttg.local_alloc %values : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
    // CHECK: ttg.async_wait
    // CHECK-NEXT: {{.*}}arith.addi
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: {{.*}}ttg.local_load
    ttg.async_wait {num = 1 : i32, tle.pipe_wait}
    %sum = arith.addi %a, %a : i32
    %data = ttg.local_load %buf : !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipe_wait_and_explicit_sync
  tt.func @pipe_wait_and_explicit_sync(%values: tensor<128xf32, #blocked>, %a: i32) {
    %buf = ttg.local_alloc %values : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
    // CHECK: ttg.async_wait
    // CHECK-NEXT: {{.*}}arith.addi
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: {{.*}}ttg.local_load
    ttg.async_wait {num = 1 : i32, tle.pipe_wait}
    %sum = arith.addi %a, %a : i32
    ttg.local_barrier
    %data = ttg.local_load %buf : !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @ordinary_wait_unchanged
  tt.func @ordinary_wait_unchanged(%values: tensor<128xf32, #blocked>, %a: i32) {
    %buf = ttg.local_alloc %values : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
    // CHECK: ttg.async_wait
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: {{.*}}arith.addi
    ttg.async_wait {num = 1 : i32}
    %sum = arith.addi %a, %a : i32
    %data = ttg.local_load %buf : !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

// A wait on only one branch must not clear the other branch's dependency.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @conditional_wait
  tt.func @conditional_wait(%values: tensor<128xf32, #blocked>, %condition: i1) {
    %id = arith.constant 3 : i32
    %threads = arith.constant 128 : i32
    %buf = ttg.local_alloc %values : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
    scf.if %condition {
      ttng.wait_barrier_named %id, %threads : i32, i32
    }
    // CHECK: ttg.local_barrier
    // CHECK-NEXT: {{.*}}ttg.local_load
    %data = ttg.local_load %buf : !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @dynamic_participant_count
  tt.func @dynamic_participant_count(%values: tensor<128xf32, #blocked>, %threads: i32) {
    %id = arith.constant 3 : i32
    %buf = ttg.local_alloc %values : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
    // CHECK: ttng.wait_barrier_named
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: {{.*}}ttg.local_load
    ttng.wait_barrier_named %id, %threads : i32, i32
    %data = ttg.local_load %buf : !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable> -> tensor<128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @other_target_pipe_wait
  tt.func @other_target_pipe_wait(%values: tensor<128xf32, #blocked>, %a: i32) {
    %buf = ttg.local_alloc %values : (tensor<128xf32, #blocked>) -> !ttg.memdesc<128xf32, #shared, #ttg.shared_memory, mutable>
    // CHECK: ttg.async_wait
    // CHECK-NEXT: ttg.local_barrier
    // CHECK-NEXT: {{.*}}arith.addi
    ttg.async_wait {num = 1 : i32, tle.pipe_wait}
    %sum = arith.addi %a, %a : i32
    tt.return
  }
}
