// RUN: triton-opt %s -split-input-file --tritonppu-tle-lower-tma-copy -verify-diagnostics

// A descriptor copy moves data between one tensor descriptor and one shared
// buffer. A copy with a shared buffer on both sides names no descriptor.

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.target" = "ppu:80", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @copy_between_two_buffers(%a: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    // expected-error @below {{expects one tensor descriptor and one shared buffer operand}}
    ttg.tma_copy %a, %b, [%c0, %c0] : !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, !ttg.memdesc<64x64xf16, #shared, #smem, mutable>
    tt.return
  }
}
