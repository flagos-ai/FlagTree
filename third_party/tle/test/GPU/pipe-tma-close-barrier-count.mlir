// RUN: triton-opt %s -triton-tle-lower-pipe-to-nvws -nvgpu-test-ws-lower-token | FileCheck %s

#stages = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32, rank = 3}>
#tile = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // A TMA data commit contributes one arrival, including when the same pipe
  // later publishes a close tag. The close must not replace this with 128.
  // CHECK-LABEL: tt.func @tma_close_keeps_payload_count
  // CHECK: ttng.init_barrier {{.*}}, 1 :
  // CHECK: ttng.init_barrier {{.*}}, 128 :
  // CHECK: ttng.init_barrier {{.*}}, 1 :
  // CHECK: ttng.init_barrier {{.*}}, 128 :
  // CHECK: ttng.barrier_expect {{.*}}, 8192
  // CHECK: ttng.async_tma_copy_global_to_local
  // CHECK: ttng.arrive_barrier {{.*}}, 1
  tt.func @tma_close_keeps_payload_count(%a: !ttg.memdesc<2x32x64xf32, #stages, #smem, mutable>, %desc: !tt.tensordesc<tensor<32x64xf32, #tile>>) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #stages, #smem, mutable>
    tle.pipe.writer_acquire %a[%zero, %false] {capacity = 2 : i32, pipe_name = "tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #stages, #smem, mutable>
    %slot = ttg.memdesc_index %a[%zero] : !ttg.memdesc<2x32x64xf32, #stages, #smem, mutable> -> !ttg.memdesc<32x64xf32, #tile, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%zero, %zero] : !tt.tensordesc<tensor<32x64xf32, #tile>>, !ttg.memdesc<32x64xf32, #tile, #smem, mutable>
    tle.pipe.writer_commit %a[%zero] {capacity = 2 : i32, pipe_name = "tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #stages, #smem, mutable>
    tle.pipe.writer_close %a[%one, %false] {capacity = 2 : i32, pipe_name = "tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #stages, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%one, %false] {capacity = 2 : i32, pipe_name = "tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #stages, #smem, mutable>
    scf.if %closed {}
    tle.pipe.reader_release %a[%one] {capacity = 2 : i32, pipe_name = "tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #stages, #smem, mutable>
    tt.return
  }

}
