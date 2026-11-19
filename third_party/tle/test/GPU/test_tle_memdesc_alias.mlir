// RUN: triton-opt %s -split-input-file --verify-diagnostics | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @valid_memdesc_alias
  tt.func @valid_memdesc_alias(%arg0: tensor<64x64xbf16, #blocked>) {
    %src = ttg.local_alloc %arg0 : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    // CHECK: tle.memdesc_alias
    // CHECK-SAME: offset_bytes = 128 : i64
    %alias = tle.memdesc_alias %src {offset_bytes = 128 : i64} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<32x64xbf16, #shared1, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_negative_offset(%src: !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{expects non-negative offset_bytes}}
    %alias = tle.memdesc_alias %src {offset_bytes = -1 : i64} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<1x1xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_unaligned_offset(%src: !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{expects offset_bytes to be aligned}}
    %alias = tle.memdesc_alias %src {offset_bytes = 1 : i64} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<1x1xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_out_of_bounds(%src: !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{result byte range must fit within the source view}}
    %alias = tle.memdesc_alias %src {offset_bytes = 8192 : i64} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<1x1xbf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_mutable_alias_from_immutable(%src: !ttg.memdesc<64x64xbf16, #shared, #smem>) {
    // expected-error @+1 {{cannot create a mutable alias from an immutable source}}
    %alias = tle.memdesc_alias %src {offset_bytes = 0 : i64} : !ttg.memdesc<64x64xbf16, #shared, #smem> -> !ttg.memdesc<1x1xbf16, #shared, #smem, mutable>
    tt.return
  }
}
