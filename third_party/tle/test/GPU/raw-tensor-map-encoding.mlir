// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// Raw TLE descriptors preserve their shared-memory contract at native lowering.
// RUN: triton-opt %s -split-input-file --triton-nvidia-optimize-descriptor-encoding --triton-tle-lower-tma-copy | FileCheck %s
#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func public @marked_raw_rowmajor
  tt.func public @marked_raw_rowmajor(%raw: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    // CHECK-NOT: ttg.tma_copy
    // CHECK: ttg.memdesc_reinterpret
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttg.tma_copy
    ttg.tma_copy %desc, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    tt.return
  }
}

// -----

#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func public @marked_raw_multiple_uses
  tt.func public @marked_raw_multiple_uses(%raw: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %a = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    // CHECK-NOT: ttg.tma_copy
    // CHECK: ttg.memdesc_reinterpret
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttg.tma_copy
    ttg.tma_copy %desc, %a, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    // CHECK: ttg.memdesc_reinterpret
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttg.tma_copy
    ttg.tma_copy %desc, %b, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    tt.return
  }
}

// -----

#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func public @marked_raw_loop
  tt.func public @marked_raw_loop(%raw: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter = %desc) -> (!tt.tensordesc<tensor<64x256xbf16>>) : i32 {
    // CHECK-NOT: ttg.tma_copy
    // CHECK: ttg.memdesc_reinterpret
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttg.tma_copy
    ttg.tma_copy %iter, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
      scf.yield %iter : !tt.tensordesc<tensor<64x256xbf16>>
    }
    // CHECK: ttg.memdesc_reinterpret
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttg.tma_copy
    ttg.tma_copy %result, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    tt.return
  }
}

// -----

#nv128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func public @unmarked_raw_native_default
  tt.func public @unmarked_raw_native_default(%raw: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #nv128, #smem, mutable>
    // CHECK-NOT: ttg.tma_copy
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttg.tma_copy
    ttg.tma_copy %desc, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #nv128, #smem, mutable>
    tt.return
  }
}

// -----

#nv128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func public @unmarked_public_native_default
  tt.func public @unmarked_public_native_default(%desc: !tt.tensordesc<tensor<64x256xbf16>>) {
    %c0 = arith.constant 0 : i32
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #nv128, #smem, mutable>
    // CHECK-NOT: ttg.tma_copy
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttg.tma_copy
    ttg.tma_copy %desc, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #nv128, #smem, mutable>
    tt.return
  }
}

// -----

#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func public @while_private
  tt.func public @while_private(%raw: !tt.ptr<i8>, %public_desc: !tt.tensordesc<tensor<64x256xbf16>>, %keep_going: i1) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    %result = scf.while (%before = %desc) : (!tt.tensordesc<tensor<64x256xbf16>>) -> !tt.tensordesc<tensor<64x256xbf16>> {
      scf.condition(%keep_going) %before : !tt.tensordesc<tensor<64x256xbf16>>
    } do {
    ^bb0(%body: !tt.tensordesc<tensor<64x256xbf16>>):
    // CHECK-NOT: ttg.tma_copy
    // CHECK: ttg.memdesc_reinterpret
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttg.tma_copy
      ttg.tma_copy %body, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
      scf.yield %body : !tt.tensordesc<tensor<64x256xbf16>>
    }
    // CHECK-NOT: ttg.tma_copy
    // CHECK: ttg.memdesc_reinterpret
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttg.tma_copy
    ttg.tma_copy %result, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    tt.return
  }
}

// -----
#root = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func public @marked_raw_dynamic_stage
  tt.func public @marked_raw_dynamic_stage(%raw: !tt.ptr<i8>, %slot: i32) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %storage = ttg.local_alloc : () -> !ttg.memdesc<3x64x256xbf16, #root, #smem, mutable>
    // CHECK: ttg.memdesc_index
    %dst = ttg.memdesc_index %storage[%slot] : !ttg.memdesc<3x64x256xbf16, #root, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    // CHECK: ttg.memdesc_reinterpret
    // CHECK-NOT: ttg.tma_copy
    // CHECK: ttng.async_tma_copy_global_to_local
    // CHECK-NOT: ttg.tma_copy
    ttg.tma_copy %desc, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    tt.return
  }
}
