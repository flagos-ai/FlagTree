// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s --triton-tle-optimize-local-pointer-async-stores | FileCheck %s

// #1586: logical copy planning extracts the masked load's zero fill before
// layouts are assigned. Prove zero through either wrapper order and repeated
// extraction, without accepting nonzero, negative-zero, or dynamic fills.
#a = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#b = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 2, perPhase = 1, maxPhase = 1, order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @extracted_zero
  // CHECK: ttg.async_copy_global_to_local
  // CHECK-NOT: tt.store
  // CHECK: tt.return
  tt.func @extracted_zero(%src: tensor<128x!tt.ptr<bf16>, #a>, %mask: tensor<128xi1, #a>) {
    %index = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<256xbf16, #a>
    %tile = "tle.extract_tile"(%zero, %index) {tile_shape = array<i64: 128>, tle.planned_tile} : (tensor<256xbf16, #a>, i32) -> tensor<128xbf16, #a>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #a>
    %ptr = "tle.local_pointers"(%arena, %indices) : (!ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>, tensor<128xi32, #a>) -> tensor<128x!tt.ptr<bf16, 3>, #a>
    %value = tt.load %src, %mask, %tile : tensor<128x!tt.ptr<bf16>, #a>
    tt.store %ptr, %value : tensor<128x!tt.ptr<bf16, 3>, #a>
    tt.return
  }

  // CHECK-LABEL: @wrapped_zero
  // CHECK: ttg.async_copy_global_to_local
  // CHECK-NOT: tt.store
  // CHECK: tt.return
  tt.func @wrapped_zero(%src: tensor<128x!tt.ptr<bf16>, #b>, %mask: tensor<128xi1, #b>) {
    %index = arith.constant 1 : i32
    %zero = arith.constant dense<0.0> : tensor<256xbf16, #a>
    %tile = "tle.extract_tile"(%zero, %index) {tile_shape = array<i64: 128>, tle.planned_tile} : (tensor<256xbf16, #a>, i32) -> tensor<128xbf16, #a>
    %other = ttg.convert_layout %tile : tensor<128xbf16, #a> -> tensor<128xbf16, #b>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #b>
    %ptr = "tle.local_pointers"(%arena, %indices) : (!ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>, tensor<128xi32, #b>) -> tensor<128x!tt.ptr<bf16, 3>, #b>
    %value = tt.load %src, %mask, %other : tensor<128x!tt.ptr<bf16>, #b>
    tt.store %ptr, %value : tensor<128x!tt.ptr<bf16, 3>, #b>
    tt.return
  }

  // CHECK-LABEL: @nested_integer_zero
  // CHECK: ttg.async_copy_global_to_local
  // CHECK-NOT: tt.store
  // CHECK: tt.return
  tt.func @nested_integer_zero(%src: tensor<128x!tt.ptr<i32>, #b>, %mask: tensor<128xi1, #b>) {
    %index = arith.constant 1 : i32
    %zero = arith.constant dense<0> : tensor<512xi32, #a>
    %converted = ttg.convert_layout %zero : tensor<512xi32, #a> -> tensor<512xi32, #b>
    %first = "tle.extract_tile"(%converted, %index) {tile_shape = array<i64: 256>, tle.planned_tile} : (tensor<512xi32, #b>, i32) -> tensor<256xi32, #b>
    %middle = ttg.convert_layout %first : tensor<256xi32, #b> -> tensor<256xi32, #a>
    %second = "tle.extract_tile"(%middle, %index) {tile_shape = array<i64: 128>, tle.planned_tile} : (tensor<256xi32, #a>, i32) -> tensor<128xi32, #a>
    %other = ttg.convert_layout %second : tensor<128xi32, #a> -> tensor<128xi32, #b>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #shared, #ttg.shared_memory, mutable>
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #b>
    %ptr = "tle.local_pointers"(%arena, %indices) : (!ttg.memdesc<128xi32, #shared, #ttg.shared_memory, mutable>, tensor<128xi32, #b>) -> tensor<128x!tt.ptr<i32, 3>, #b>
    %value = tt.load %src, %mask, %other : tensor<128x!tt.ptr<i32>, #b>
    tt.store %ptr, %value : tensor<128x!tt.ptr<i32, 3>, #b>
    tt.return
  }

  // CHECK-LABEL: @wrapped_nonzero
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.return
  tt.func @wrapped_nonzero(%src: tensor<128x!tt.ptr<bf16>, #b>, %mask: tensor<128xi1, #b>) {
    %index = arith.constant 1 : i32
    %zero = arith.constant dense<1.0> : tensor<256xbf16, #a>
    %tile = "tle.extract_tile"(%zero, %index) {tile_shape = array<i64: 128>, tle.planned_tile} : (tensor<256xbf16, #a>, i32) -> tensor<128xbf16, #a>
    %other = ttg.convert_layout %tile : tensor<128xbf16, #a> -> tensor<128xbf16, #b>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #b>
    %ptr = "tle.local_pointers"(%arena, %indices) : (!ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>, tensor<128xi32, #b>) -> tensor<128x!tt.ptr<bf16, 3>, #b>
    %value = tt.load %src, %mask, %other : tensor<128x!tt.ptr<bf16>, #b>
    tt.store %ptr, %value : tensor<128x!tt.ptr<bf16, 3>, #b>
    tt.return
  }

  // CHECK-LABEL: @wrapped_negative_zero
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.return
  tt.func @wrapped_negative_zero(%src: tensor<128x!tt.ptr<bf16>, #b>, %mask: tensor<128xi1, #b>) {
    %index = arith.constant 1 : i32
    %zero = arith.constant dense<-0.0> : tensor<256xbf16, #a>
    %tile = "tle.extract_tile"(%zero, %index) {tile_shape = array<i64: 128>, tle.planned_tile} : (tensor<256xbf16, #a>, i32) -> tensor<128xbf16, #a>
    %other = ttg.convert_layout %tile : tensor<128xbf16, #a> -> tensor<128xbf16, #b>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #b>
    %ptr = "tle.local_pointers"(%arena, %indices) : (!ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>, tensor<128xi32, #b>) -> tensor<128x!tt.ptr<bf16, 3>, #b>
    %value = tt.load %src, %mask, %other : tensor<128x!tt.ptr<bf16>, #b>
    tt.store %ptr, %value : tensor<128x!tt.ptr<bf16, 3>, #b>
    tt.return
  }

  // CHECK-LABEL: @wrapped_dynamic
  // CHECK-NOT: ttg.async_copy_global_to_local
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.return
  tt.func @wrapped_dynamic(%src: tensor<128x!tt.ptr<bf16>, #b>, %mask: tensor<128xi1, #b>, %zero: tensor<256xbf16, #a>) {
    %index = arith.constant 1 : i32
    %tile = "tle.extract_tile"(%zero, %index) {tile_shape = array<i64: 128>, tle.planned_tile} : (tensor<256xbf16, #a>, i32) -> tensor<128xbf16, #a>
    %other = ttg.convert_layout %tile : tensor<128xbf16, #a> -> tensor<128xbf16, #b>
    %arena = ttg.local_alloc : () -> !ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>
    %indices = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32, #b>
    %ptr = "tle.local_pointers"(%arena, %indices) : (!ttg.memdesc<128xbf16, #shared, #ttg.shared_memory, mutable>, tensor<128xi32, #b>) -> tensor<128x!tt.ptr<bf16, 3>, #b>
    %value = tt.load %src, %mask, %other : tensor<128x!tt.ptr<bf16>, #b>
    tt.store %ptr, %value : tensor<128x!tt.ptr<bf16, 3>, #b>
    tt.return
  }
}
