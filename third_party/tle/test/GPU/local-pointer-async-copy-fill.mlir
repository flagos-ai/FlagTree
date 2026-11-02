// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s --triton-tle-optimize-local-pointer-async-stores | FileCheck %s

// Layout conversion preserves the fill bits. cp.async zero-fills bytes, so
// positive floating zero and integer zero are legal through conversions;
// negative floating zero and nonzero fills must retain the load/store pair.
#a = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#b = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @converted_positive_zero
  // CHECK: ttg.async_copy_global_to_local
  // CHECK-NOT: tt.store
  // CHECK: tt.return
  tt.func @converted_positive_zero(%src: tensor<128x!tt.ptr<bf16>, #b>, %mask: tensor<128xi1, #b>) {
    %zero = arith.constant dense<0.0> : tensor<128xbf16, #a>
    %other = ttg.convert_layout %zero : tensor<128xbf16, #a> -> tensor<128xbf16, #b>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #b>
    %ptr = "tle.local_pointers"(%arena, %indices) : (!ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>, tensor<128xi32, #b>) -> tensor<128x!tt.ptr<bf16, 3>, #b>
    %value = tt.load %src, %mask, %other : tensor<128x!tt.ptr<bf16>, #b>
    tt.store %ptr, %value : tensor<128x!tt.ptr<bf16, 3>, #b>
    tt.return
  }
  // CHECK-LABEL: @converted_integer_zero
  // CHECK: ttg.async_copy_global_to_local
  // CHECK-NOT: tt.store
  // CHECK: tt.return
  tt.func @converted_integer_zero(%src: tensor<128x!tt.ptr<i32>, #b>, %mask: tensor<128xi1, #b>) {
    %zero = arith.constant dense<0> : tensor<128xi32, #a>
    %other = ttg.convert_layout %zero : tensor<128xi32, #a> -> tensor<128xi32, #b>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #ttg.shared_memory, mutable>
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #b>
    %ptr = "tle.local_pointers"(%arena, %indices) : (!ttg.memdesc<128xi32, #shared, #ttg.shared_memory, mutable>, tensor<128xi32, #b>) -> tensor<128x!tt.ptr<i32, 3>, #b>
    %value = tt.load %src, %mask, %other : tensor<128x!tt.ptr<i32>, #b>
    tt.store %ptr, %value : tensor<128x!tt.ptr<i32, 3>, #b>
    tt.return
  }
  // CHECK-LABEL: @negative_zero
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.return
  tt.func @negative_zero(%src: tensor<128x!tt.ptr<bf16>, #b>, %mask: tensor<128xi1, #b>) {
    %other = arith.constant dense<-0.0> : tensor<128xbf16, #b>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #b>
    %ptr = "tle.local_pointers"(%arena, %indices) : (!ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>, tensor<128xi32, #b>) -> tensor<128x!tt.ptr<bf16, 3>, #b>
    %value = tt.load %src, %mask, %other : tensor<128x!tt.ptr<bf16>, #b>
    tt.store %ptr, %value : tensor<128x!tt.ptr<bf16, 3>, #b>
    tt.return
  }
  // CHECK-LABEL: @converted_nonzero
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.return
  tt.func @converted_nonzero(%src: tensor<128x!tt.ptr<bf16>, #b>, %mask: tensor<128xi1, #b>) {
    %one = arith.constant dense<1.0> : tensor<128xbf16, #a>
    %other = ttg.convert_layout %one : tensor<128xbf16, #a> -> tensor<128xbf16, #b>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #b>
    %ptr = "tle.local_pointers"(%arena, %indices) : (!ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>, tensor<128xi32, #b>) -> tensor<128x!tt.ptr<bf16, 3>, #b>
    %value = tt.load %src, %mask, %other : tensor<128x!tt.ptr<bf16>, #b>
    tt.store %ptr, %value : tensor<128x!tt.ptr<bf16, 3>, #b>
    tt.return
  }
}
