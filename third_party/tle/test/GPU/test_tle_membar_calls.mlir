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
// RUN: triton-opt %s -split-input-file --allocate-shared-memory -test-tritonamdgpu-membar | FileCheck %s

// Membar ordering across tt.call: a call is ordered against what the callee
// touches before its first CTA barrier, translated to the call frame offset.

#blockedLarge = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#sharedLarge = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#blockedCallSrc = #ttg.blocked<{sizePerThread = [1, 32], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#mmaCall = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 8]}>
#blockedCallDst = #ttg.dot_op<{opIdx = 0, parent = #mmaCall, kWidth = 2}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @callee_call_offset_membar() -> tensor<128x32xf16, #blockedCallDst> {
    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    tt.return %cvt : tensor<128x32xf16, #blockedCallDst>
  }

  // The call's virtual buffer is offset by the large allocation. The
  // subsequent scratch op should alias at the same offset and require a membar.
  // CHECK-LABEL: @caller_call_offset_membar
  // CHECK: tt.call @callee_call_offset_membar{{.*}}allocation.offset = [[CALL_OFFSET:[1-9][0-9]*]]
  // CHECK: ttg.local_barrier
  // CHECK-NEXT: ttg.convert_layout{{.*}}allocation.offset = [[CALL_OFFSET]]
  tt.func @caller_call_offset_membar() -> tensor<128x32xf16, #blockedCallDst> {
    %large = arith.constant dense<0> : tensor<65536xi8, #blockedLarge>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<65536xi8, #sharedLarge, #smem, mutable>
    ttg.local_store %large, %buf : tensor<65536xi8, #blockedLarge> -> !ttg.memdesc<65536xi8, #sharedLarge, #smem, mutable>

    %call = tt.call @callee_call_offset_membar() : () -> tensor<128x32xf16, #blockedCallDst>

    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    %sum = arith.addf %call, %cvt : tensor<128x32xf16, #blockedCallDst>

    %ld = ttg.local_load %buf : !ttg.memdesc<65536xi8, #sharedLarge, #smem, mutable> -> tensor<65536xi8, #blockedLarge>
    ttg.local_dealloc %buf : !ttg.memdesc<65536xi8, #sharedLarge, #smem, mutable>
    tt.return %sum : tensor<128x32xf16, #blockedCallDst>
  }

  tt.func private @callee_writes_first() -> tensor<128x32xf16, #blockedCallDst> {
    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    tt.return %cvt : tensor<128x32xf16, #blockedCallDst>
  }

  // The caller's conversion leaves a pending shared-memory read. The callee
  // starts with a conversion that writes the same call frame, so a barrier is
  // required immediately before the call.
  // CHECK-LABEL: @caller_pending_read_then_call
  // CHECK: ttg.convert_layout
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: tt.call @callee_writes_first
  tt.func @caller_pending_read_then_call() -> tensor<128x32xf16, #blockedCallDst> {
    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    %call = tt.call @callee_writes_first() : () -> tensor<128x32xf16, #blockedCallDst>
    %sum = arith.addf %call, %cvt : tensor<128x32xf16, #blockedCallDst>
    tt.return %sum : tensor<128x32xf16, #blockedCallDst>
  }

  tt.func private @callee_writes_then_barrier() -> tensor<128x32xf16, #blockedCallDst> {
    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    ttg.local_barrier
    tt.return %cvt : tensor<128x32xf16, #blockedCallDst>
  }

  // A trailing callee barrier cannot order the callee's first write against
  // the caller's access before the call.
  // CHECK-LABEL: @caller_pending_read_then_callee_trailing_barrier
  // CHECK: ttg.convert_layout
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: tt.call @callee_writes_then_barrier
  tt.func @caller_pending_read_then_callee_trailing_barrier() -> tensor<128x32xf16, #blockedCallDst> {
    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    %call = tt.call @callee_writes_then_barrier() : () -> tensor<128x32xf16, #blockedCallDst>
    %sum = arith.addf %call, %cvt : tensor<128x32xf16, #blockedCallDst>
    tt.return %sum : tensor<128x32xf16, #blockedCallDst>
  }

  tt.func private @callee_barrier_then_writes() -> tensor<128x32xf16, #blockedCallDst> {
    ttg.local_barrier
    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    tt.return %cvt : tensor<128x32xf16, #blockedCallDst>
  }

  // A leading callee barrier synchronizes the caller's pending access.
  // CHECK-LABEL: @callee_entry_barrier_covers_caller
  // CHECK: ttg.convert_layout
  // CHECK-NEXT: tt.call @callee_barrier_then_writes
  tt.func @callee_entry_barrier_covers_caller() -> tensor<128x32xf16, #blockedCallDst> {
    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    %call = tt.call @callee_barrier_then_writes() : () -> tensor<128x32xf16, #blockedCallDst>
    %sum = arith.addf %call, %cvt : tensor<128x32xf16, #blockedCallDst>
    tt.return %sum : tensor<128x32xf16, #blockedCallDst>
  }

  tt.func private @nested_entry_c() -> tensor<128x32xf16, #blockedCallDst> {
    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    tt.return %cvt : tensor<128x32xf16, #blockedCallDst>
  }

  // CHECK-LABEL: @nested_entry_b
  // CHECK-NOT: ttg.local_barrier
  // CHECK: tt.call @nested_entry_c
  tt.func private @nested_entry_b() -> tensor<128x32xf16, #blockedCallDst> {
    %call = tt.call @nested_entry_c() : () -> tensor<128x32xf16, #blockedCallDst>
    tt.return %call : tensor<128x32xf16, #blockedCallDst>
  }

  // The first call has no pending shared-memory accesses and needs no barrier.
  // The conversion before the second call conflicts with the entry state
  // propagated through B from C.
  // CHECK-LABEL: @nested_entry_a
  // CHECK-NOT: ttg.local_barrier
  // CHECK: %[[FIRST:.*]] = tt.call @nested_entry_b
  // CHECK: ttg.convert_layout
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: %[[SECOND:.*]] = tt.call @nested_entry_b
  tt.func @nested_entry_a() -> tensor<128x32xf16, #blockedCallDst> {
    %first = tt.call @nested_entry_b() : () -> tensor<128x32xf16, #blockedCallDst>
    %cst = arith.constant dense<0.0> : tensor<128x32xf16, #blockedCallSrc>
    %cvt = ttg.convert_layout %cst : tensor<128x32xf16, #blockedCallSrc> -> tensor<128x32xf16, #blockedCallDst>
    %second = tt.call @nested_entry_b() : () -> tensor<128x32xf16, #blockedCallDst>
    %sum0 = arith.addf %first, %cvt : tensor<128x32xf16, #blockedCallDst>
    %sum1 = arith.addf %sum0, %second : tensor<128x32xf16, #blockedCallDst>
    tt.return %sum1 : tensor<128x32xf16, #blockedCallDst>
  }
}

// -----

#blockedArrive = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#sharedArrive = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func private @arrive_then_scratch(%barrier: !ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>) -> tensor<128xi32, #blockedArrive> {
    ttng.arrive_barrier %barrier, 128 {release_fence = true} : !ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>
    %zero = arith.constant dense<0> : tensor<128xi32, #blockedArrive>
    %scratch = ttg.local_alloc %zero : (tensor<128xi32, #blockedArrive>) -> !ttg.memdesc<128xi32, #sharedArrive, #smem, mutable>
    %result = ttg.local_load %scratch : !ttg.memdesc<128xi32, #sharedArrive, #smem, mutable> -> tensor<128xi32, #blockedArrive>
    tt.return %result : tensor<128xi32, #blockedArrive>
  }

  // An elected arrive at the callee's entry publishes all shared memory,
  // including the caller's data below the callee frame, so the caller's store
  // needs a CTA barrier before the call even though the frame is offset.
  // CHECK-LABEL: @caller_nonzero_frame_elected_arrive
  // CHECK: ttg.local_store
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: tt.call @arrive_then_scratch{{.*}}allocation.offset = {{[1-9][0-9]*}}
  tt.func @caller_nonzero_frame_elected_arrive(%values: tensor<128xi32, #blockedArrive>) -> tensor<128xi32, #blockedArrive> {
    %data = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #sharedArrive, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>
    ttng.init_barrier %barrier, 128 : !ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>
    gpu.barrier
    ttg.local_store %values, %data : tensor<128xi32, #blockedArrive> -> !ttg.memdesc<128xi32, #sharedArrive, #smem, mutable>
    %tmp = tt.call @arrive_then_scratch(%barrier) : (!ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>) -> tensor<128xi32, #blockedArrive>
    %loaded = ttg.local_load %data : !ttg.memdesc<128xi32, #sharedArrive, #smem, mutable> -> tensor<128xi32, #blockedArrive>
    %sum = arith.addi %loaded, %tmp : tensor<128xi32, #blockedArrive>
    tt.return %sum : tensor<128xi32, #blockedArrive>
  }

  tt.func private @forward_to_arrive(%barrier: !ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>) -> tensor<128xi32, #blockedArrive> {
    %result = tt.call @arrive_then_scratch(%barrier) : (!ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>) -> tensor<128xi32, #blockedArrive>
    tt.return %result : tensor<128xi32, #blockedArrive>
  }

  // The all-memory interval must survive translation through every call level.
  // CHECK-LABEL: @caller_nonzero_frame_nested_elected_arrive
  // CHECK: ttg.local_store
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: tt.call @forward_to_arrive{{.*}}allocation.offset = {{[1-9][0-9]*}}
  tt.func @caller_nonzero_frame_nested_elected_arrive(%values: tensor<128xi32, #blockedArrive>) -> tensor<128xi32, #blockedArrive> {
    %data = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #sharedArrive, #smem, mutable>
    %barrier = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>
    ttng.init_barrier %barrier, 128 : !ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>
    gpu.barrier
    ttg.local_store %values, %data : tensor<128xi32, #blockedArrive> -> !ttg.memdesc<128xi32, #sharedArrive, #smem, mutable>
    %tmp = tt.call @forward_to_arrive(%barrier) : (!ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>) -> tensor<128xi32, #blockedArrive>
    %loaded = ttg.local_load %data : !ttg.memdesc<128xi32, #sharedArrive, #smem, mutable> -> tensor<128xi32, #blockedArrive>
    %sum = arith.addi %loaded, %tmp : tensor<128xi32, #blockedArrive>
    tt.return %sum : tensor<128xi32, #blockedArrive>
  }

  // Control: with the callee frame at offset 0 the barrier was always kept.
  // CHECK-LABEL: @caller_zero_frame_elected_arrive
  // CHECK: ttg.local_store
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: tt.call @arrive_then_scratch{{.*}}allocation.offset = 0
  tt.func @caller_zero_frame_elected_arrive(%values: tensor<128xi32, #blockedArrive>, %barrier: !ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>) -> tensor<128xi32, #blockedArrive> {
    %data = ttg.local_alloc : () -> !ttg.memdesc<128xi32, #sharedArrive, #smem, mutable>
    ttg.local_store %values, %data : tensor<128xi32, #blockedArrive> -> !ttg.memdesc<128xi32, #sharedArrive, #smem, mutable>
    %tmp = tt.call @arrive_then_scratch(%barrier) : (!ttg.memdesc<1xi64, #sharedArrive, #smem, mutable>) -> tensor<128xi32, #blockedArrive>
    tt.return %tmp : tensor<128xi32, #blockedArrive>
  }
}
