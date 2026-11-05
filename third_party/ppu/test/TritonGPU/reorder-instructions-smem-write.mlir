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

// RUN: triton-opt %s -split-input-file -tritongpu-reorder-instructions=keep-smem-read-order=true | FileCheck %s --check-prefixes=COMMON,ON
// RUN: triton-opt %s -split-input-file -tritongpu-reorder-instructions | FileCheck %s --check-prefixes=COMMON,OFF

// With the option (FLAGTREE_KEEP_SMEM_READ_ORDER), a load of mutable shared
// memory is not sunk into a loop past anything that may write the buffer it
// reads: a write through a descriptor of the same allocation (through views),
// a store through a shared pointer, or an op whose effects are unknown. A
// write to another allocation or to global memory does not hold it back.
// Without the option the load is sunk, as upstream does.

// COMMON-LABEL: no_sink_load_past_write
//       ON: ttg.local_load
//       ON: scf.for
//       ON:   ttg.local_store
//       ON:   ttg.local_load
//   ON-NOT:   ttg.local_load
//       ON:   tt.dot
//  OFF-NOT: ttg.local_load
//      OFF: scf.for
//      OFF:   ttg.local_store
//      OFF:   ttg.local_load
//      OFF:   ttg.local_load
//      OFF:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_sink_load_past_write(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %S = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %AD = ttg.local_load %S : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      ttg.local_store %B, %S : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
      %BD = ttg.local_load %S : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %d = tt.dot %AD, %BD, %acc, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    %o = ttg.convert_layout %r : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %o : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// Without a write in between, the load is still sunk into the loop.
// COMMON-LABEL: sink_load_into_loop
//   COMMON-NOT: ttg.local_load
//       COMMON: scf.for
//       COMMON:   ttg.local_load
//       COMMON:   ttg.local_load
//       COMMON:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_load_into_loop(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %S = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %AD = ttg.local_load %S : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
      %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %d = tt.dot %AD, %BD, %acc, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    %o = ttg.convert_layout %r : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %o : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A write to another allocation does not change what the load reads: the load
// is sunk either way.
// COMMON-LABEL: sink_load_past_write_of_other_alloc
//   COMMON-NOT: ttg.local_load
//       COMMON: scf.for
//       COMMON:   ttg.local_store
//       COMMON:   ttg.local_load
//       COMMON:   ttg.local_load
//       COMMON:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_load_past_write_of_other_alloc(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %S = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %T = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %AD = ttg.local_load %S : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      ttg.local_store %B, %T : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
      %BD = ttg.local_load %T : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %d = tt.dot %AD, %BD, %acc, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    %o = ttg.convert_layout %r : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %o : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// The load reads one slot of a buffer and the loop writes a slot of the same
// buffer through a view: the same allocation, so the load stays.
// COMMON-LABEL: no_sink_load_past_write_through_view
//       ON: ttg.local_load
//       ON: scf.for
//       ON:   ttg.local_store
//       ON:   ttg.local_load
//   ON-NOT:   ttg.local_load
//       ON:   tt.dot
//  OFF-NOT: ttg.local_load
//      OFF: scf.for
//      OFF:   ttg.local_store
//      OFF:   ttg.local_load
//      OFF:   ttg.local_load
//      OFF:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_sink_load_past_write_through_view(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %S = ttg.local_alloc : () -> !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable>
    %S0 = ttg.memdesc_index %S[%c0] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    ttg.local_store %A, %S0 : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %AD = ttg.local_load %S0 : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      %Si = ttg.memdesc_index %S[%i] : !ttg.memdesc<2x32x32xf32, #shared, #smem, mutable> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
      ttg.local_store %B, %Si : tensor<32x32xf32, #blocked> -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
      %BD = ttg.local_load %Si : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %d = tt.dot %AD, %BD, %acc, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    %o = ttg.convert_layout %r : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %o : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A store to global memory does not change what the load reads: the load is
// sunk either way.
// COMMON-LABEL: sink_load_past_global_store
//   COMMON-NOT: ttg.local_load
//       COMMON: scf.for
//       COMMON:   tt.store
//       COMMON:   ttg.local_load
//       COMMON:   ttg.local_load
//       COMMON:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @sink_load_past_global_store(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %arg1: tensor<32x32x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %S = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %AD = ttg.local_load %S : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      tt.store %arg1, %B : tensor<32x32x!tt.ptr<f32>, #blocked>
      %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
      %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %d = tt.dot %AD, %BD, %acc, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    %o = ttg.convert_layout %r : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %o : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// A store through pointers of tle.local_pointers writes shared memory; which
// buffer is not tracked through pointer arithmetic, so the load stays even
// though the pointers are of another allocation.
// COMMON-LABEL: no_sink_load_past_local_pointer_store
//       ON: ttg.local_load
//       ON: scf.for
//       ON:   tt.store
//       ON:   ttg.local_load
//   ON-NOT:   ttg.local_load
//       ON:   tt.dot
//  OFF-NOT: ttg.local_load
//      OFF: scf.for
//      OFF:   tt.store
//      OFF:   ttg.local_load
//      OFF:   ttg.local_load
//      OFF:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_sink_load_past_local_pointer_store(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %S = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %T = ttg.local_alloc : () -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %TP = "tle.local_pointers"(%T) : (!ttg.memdesc<32x32xf32, #shared, #smem, mutable>) -> tensor<32x32x!tt.ptr<f32, 3>, #blocked>
    %AD = ttg.local_load %S : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      tt.store %TP, %B : tensor<32x32x!tt.ptr<f32, 3>, #blocked>
      %BD = ttg.local_load %T : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %d = tt.dot %AD, %BD, %acc, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    %o = ttg.convert_layout %r : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %o : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// An op whose effects are unknown (gpu.barrier declares none) may write
// anything: the load stays.
// COMMON-LABEL: no_sink_load_past_unknown_effects
//       ON: ttg.local_load
//       ON: scf.for
//       ON:   gpu.barrier
//       ON:   ttg.local_load
//   ON-NOT:   ttg.local_load
//       ON:   tt.dot
//  OFF-NOT: ttg.local_load
//      OFF: scf.for
//      OFF:   gpu.barrier
//      OFF:   ttg.local_load
//      OFF:   ttg.local_load
//      OFF:   tt.dot
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 4, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @no_sink_load_past_unknown_effects(%arg0: tensor<32x32x!tt.ptr<f32>, #blocked>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x32xf32, #mma>
    %A = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
    %S = ttg.local_alloc %A : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem, mutable>
    %AD = ttg.local_load %S : !ttg.memdesc<32x32xf32, #shared, #smem, mutable> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>
    %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %cst) -> (tensor<32x32xf32, #mma>) : i32 {
      %B = tt.load %arg0 : tensor<32x32x!tt.ptr<f32>, #blocked>
      gpu.barrier
      %BS = ttg.local_alloc %B : (tensor<32x32xf32, #blocked>) -> !ttg.memdesc<32x32xf32, #shared, #smem>
      %BD = ttg.local_load %BS : !ttg.memdesc<32x32xf32, #shared, #smem> -> tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>
      %d = tt.dot %AD, %BD, %acc, inputPrecision = tf32 : tensor<32x32xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<32x32xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<32x32xf32, #mma>
      scf.yield %d : tensor<32x32xf32, #mma>
    }
    %o = ttg.convert_layout %r : tensor<32x32xf32, #mma> -> tensor<32x32xf32, #blocked>
    tt.store %arg0, %o : tensor<32x32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
