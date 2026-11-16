// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -split-input-file --triton-nvidia-optimize-descriptor-encoding --triton-tle-lower-tma-copy -verify-diagnostics
#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func public @unmarked_raw_plain_rejected
  tt.func public @unmarked_raw_plain_rejected(%raw: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    // expected-error @+1 {{requires tensor descriptor shared encoding}}
    ttg.tma_copy %desc, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    tt.return
  }
}

// -----

#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#nv128 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func public @marked_raw_conflicting_layouts
  tt.func public @marked_raw_conflicting_layouts(%raw: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %plain = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    %swizzled = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #nv128, #smem, mutable>
    // expected-error @+1 {{requires tensor descriptor shared encoding}}
    ttg.tma_copy %desc, %plain, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    ttg.tma_copy %desc, %swizzled, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #nv128, #smem, mutable>
    tt.return
  }
}

// -----

#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func public @mixed_marked_unmarked_group_rejected
  tt.func public @mixed_marked_unmarked_group_rejected(%raw: !tt.ptr<i8>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %marked = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %unmarked = ttng.reinterpret_tensor_descriptor %raw : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    %mixed = scf.if %choose -> (!tt.tensordesc<tensor<64x256xbf16>>) {
      scf.yield %marked : !tt.tensordesc<tensor<64x256xbf16>>
    } else {
      scf.yield %unmarked : !tt.tensordesc<tensor<64x256xbf16>>
    }
    // expected-error @+1 {{requires tensor descriptor shared encoding}}
    ttg.tma_copy %mixed, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    tt.return
  }
}

// -----

#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @unmarked_public_plain_rejected(%desc: !tt.tensordesc<tensor<64x256xbf16>>) {
    %c0 = arith.constant 0 : i32
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    // expected-error @+1 {{requires tensor descriptor shared encoding}}
    ttg.tma_copy %desc, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    tt.return
  }
}

// -----

#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @marked_raw_shape_rejected(%raw: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<32x256xbf16, #row, #smem, mutable>
    // expected-error @+1 {{requires tensor descriptor block shape}}
    ttg.tma_copy %desc, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<32x256xbf16, #row, #smem, mutable>
    tt.return
  }
}

// -----

#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @mixed_marked_public_group_rejected(%raw: !tt.ptr<i8>, %public_desc: !tt.tensordesc<tensor<64x256xbf16>>, %choose: i1) {
    %c0 = arith.constant 0 : i32
    %marked = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %mixed = scf.if %choose -> (!tt.tensordesc<tensor<64x256xbf16>>) {
      scf.yield %marked : !tt.tensordesc<tensor<64x256xbf16>>
    } else {
      scf.yield %public_desc : !tt.tensordesc<tensor<64x256xbf16>>
    }
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    // expected-error @+1 {{requires tensor descriptor shared encoding}}
    ttg.tma_copy %mixed, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    tt.return
  }
}

// -----

#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @while_public_backedge(%raw: !tt.ptr<i8>, %public_desc: !tt.tensordesc<tensor<64x256xbf16>>, %keep_going: i1) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    %result = scf.while (%before = %desc) : (!tt.tensordesc<tensor<64x256xbf16>>) -> !tt.tensordesc<tensor<64x256xbf16>> {
      scf.condition(%keep_going) %before : !tt.tensordesc<tensor<64x256xbf16>>
    } do {
    ^bb0(%body: !tt.tensordesc<tensor<64x256xbf16>>):
    // expected-error@+1 {{requires tensor descriptor shared encoding}}
      ttg.tma_copy %body, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
      scf.yield %public_desc : !tt.tensordesc<tensor<64x256xbf16>>
    }
    // expected-error@+1 {{requires tensor descriptor shared encoding}}
    ttg.tma_copy %result, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    tt.return
  }
}

// -----
#plain = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @marked_noncontiguous_destination(%raw: !tt.ptr<i8>, %dst: !ttg.memdesc<64x256xbf16, #plain, #smem, mutable, 64x512>) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    // expected-error@+1 {{native TMA destination view requires a contiguous complete allocation slice}}
    ttg.tma_copy %desc, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #plain, #smem, mutable, 64x512>
    tt.return
  }
}

// -----
#plain = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @marked_nonequivalent_destination(%raw: !tt.ptr<i8>, %dst: !ttg.memdesc<64x256xbf16, #plain, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    // expected-error@+1 {{native TMA destination view requires an identical physical shared-memory mapping}}
    ttg.tma_copy %desc, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #plain, #smem, mutable>
    tt.return
  }
}

// -----
#plain = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @marked_fp4_destination(%raw: !tt.ptr<i8>, %dst: !ttg.memdesc<64x256xi4, #plain, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %desc = ttng.reinterpret_tensor_descriptor %raw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xi4>>
    // expected-error@+1 {{native TMA destination view requires 8, 16, 32 or 64-bit elements; FP4 is unsupported}}
    ttg.tma_copy %desc, %dst, [%c0, %c0] : !tt.tensordesc<tensor<64x256xi4>>, !ttg.memdesc<64x256xi4, #plain, #smem, mutable>
    tt.return
  }
}

// -----
#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func @unmarked_native_plain_still_rejected(%desc: !tt.tensordesc<tensor<64x256xbf16, #row>>, %dst: !ttg.memdesc<64x256xbf16, #row, #smem, mutable>, %barrier: !ttg.memdesc<1xi64, #bar, #smem, mutable>, %x: i32, %pred: i1) {
    // expected-error@+1 {{TMA result must have NVMMA shared layout}}
    ttng.async_tma_copy_global_to_local %desc[%x, %x] %dst, %barrier, %pred : !tt.tensordesc<tensor<64x256xbf16, #row>>, !ttg.memdesc<1xi64, #bar, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    tt.return
  }
}
