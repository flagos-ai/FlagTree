// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT

// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %s

// Shared arguments have no local allocation ID. Their memory effects must
// still participate in RAW/WAR analysis, including through views and calls.
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func {{.*}}@borrowed_memdesc_raw
  // CHECK: {{st.shared|nvvm.stmatrix}}
  // CHECK: nvvm.barrier0
  // CHECK: {{ld.shared|nvvm.ldmatrix}}
  // CHECK: llvm.return
  tt.func private @borrowed_memdesc_raw(%arena: !ttg.memdesc<128xi32, #shared, #smem, mutable>, %value: tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    ttg.local_store %value, %arena : tensor<128xi32, #blocked> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    %result = ttg.local_load %arena : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #blocked>
    tt.return %result : tensor<128xi32, #blocked>
  }

  // CHECK-LABEL: llvm.func {{.*}}@borrowed_memdesc_war
  // CHECK: {{ld.shared|nvvm.ldmatrix}}
  // CHECK: nvvm.barrier0
  // CHECK: {{st.shared|nvvm.stmatrix}}
  // CHECK: llvm.return
  tt.func private @borrowed_memdesc_war(%arena: !ttg.memdesc<128xi32, #shared, #smem, mutable>, %value: tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    %result = ttg.local_load %arena : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #blocked>
    ttg.local_store %value, %arena : tensor<128xi32, #blocked> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    tt.return %result : tensor<128xi32, #blocked>
  }

  // CHECK-LABEL: llvm.func {{.*}}@borrowed_alias_raw
  // CHECK: {{st.shared|nvvm.stmatrix}}
  // CHECK: nvvm.barrier0
  // CHECK: {{ld.shared|nvvm.ldmatrix}}
  // CHECK: llvm.return
  tt.func private @borrowed_alias_raw(%arena: !ttg.memdesc<1024xi8, #shared, #smem, mutable>, %value: tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    %alias = tle.memdesc_alias %arena {offset_bytes = 128 : i64} : !ttg.memdesc<1024xi8, #shared, #smem, mutable> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    %offsets = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %pointers = "tle.local_pointers"(%alias, %offsets) : (!ttg.memdesc<128xi32, #shared, #smem, mutable>, tensor<128xi32, #blocked>) -> tensor<128x!tt.ptr<i32, 3>, #blocked>
    tt.store %pointers, %value : tensor<128x!tt.ptr<i32, 3>, #blocked>
    %result = ttg.local_load %alias : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #blocked>
    tt.return %result : tensor<128xi32, #blocked>
  }

  // CHECK-LABEL: llvm.func {{.*}}@shared_pointer_raw
  // CHECK: {{st.shared|nvvm.stmatrix}}
  // CHECK: nvvm.barrier0
  // CHECK: {{ld.shared|nvvm.ldmatrix}}
  // CHECK: llvm.return
  tt.func private @shared_pointer_raw(%pointers: tensor<128x!tt.ptr<i32, 3>, #blocked>, %value: tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    tt.store %pointers, %value : tensor<128x!tt.ptr<i32, 3>, #blocked>
    %result = tt.load %pointers : tensor<128x!tt.ptr<i32, 3>, #blocked>
    tt.return %result : tensor<128xi32, #blocked>
  }

  // Distinct formal arguments may alias the same caller-owned shared buffer.
  // CHECK-LABEL: llvm.func {{.*}}@shared_parameters_may_alias
  // CHECK: {{st.shared|nvvm.stmatrix}}
  // CHECK: nvvm.barrier0
  // CHECK: {{ld.shared|nvvm.ldmatrix}}
  // CHECK: llvm.return
  tt.func private @shared_parameters_may_alias(%dst: !ttg.memdesc<128xi32, #shared, #smem, mutable>, %src: !ttg.memdesc<128xi32, #shared, #smem, mutable>, %value: tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    ttg.local_store %value, %dst : tensor<128xi32, #blocked> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    %result = ttg.local_load %src : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #blocked>
    tt.return %result : tensor<128xi32, #blocked>
  }

  // A callee's unresolved read must also conflict with a caller's known arena.
  // CHECK-LABEL: llvm.func {{.*}}@read_borrowed
  // CHECK: {{ld.shared|nvvm.ldmatrix}}
  // CHECK-NOT: nvvm.barrier0
  // CHECK: llvm.return
  tt.func private @read_borrowed(%arena: !ttg.memdesc<128xi32, #shared, #smem, mutable>) -> tensor<128xi32, #blocked> attributes {noinline = true} {
    %result = ttg.local_load %arena : !ttg.memdesc<128xi32, #shared, #smem, mutable> -> tensor<128xi32, #blocked>
    tt.return %result : tensor<128xi32, #blocked>
  }

  // CHECK-LABEL: llvm.func {{.*}}@caller_reuses_arena
  // CHECK: llvm.call @read_borrowed
  // CHECK: nvvm.barrier0
  // CHECK: {{st.shared|nvvm.stmatrix}}
  // CHECK: llvm.return
  tt.func private @caller_reuses_arena(%value: tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    %result = tt.call @read_borrowed(%arena) : (!ttg.memdesc<128xi32, #shared, #smem, mutable>) -> tensor<128xi32, #blocked>
    ttg.local_store %value, %arena : tensor<128xi32, #blocked> -> !ttg.memdesc<128xi32, #shared, #smem, mutable>
    tt.return %result : tensor<128xi32, #blocked>
  }

  // Global pointers must not become unknown shared-memory accesses.
  // CHECK-LABEL: llvm.func {{.*}}@global_pointer_raw
  // CHECK-NOT: nvvm.barrier0
  // CHECK: llvm.return
  tt.func private @global_pointer_raw(%pointers: tensor<128x!tt.ptr<i32>, #blocked>, %value: tensor<128xi32, #blocked>) -> tensor<128xi32, #blocked> {
    tt.store %pointers, %value : tensor<128x!tt.ptr<i32>, #blocked>
    %result = tt.load %pointers : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return %result : tensor<128xi32, #blocked>
  }
}
