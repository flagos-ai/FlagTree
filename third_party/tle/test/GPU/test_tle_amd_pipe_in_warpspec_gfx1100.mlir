// RUN: triton-opt %s -triton-tle-lower-pipe-to-rdna | FileCheck %s --check-prefix=PIPE
// RUN: triton-opt %s -triton-tle-lower-pipe-to-rdna -tritonamdgpu-convert-warp-specialize | FileCheck %s --check-prefix=WS
// RUN: triton-opt %s -triton-tle-lower-pipe-to-rdna -tritonamdgpu-convert-warp-specialize -allocate-shared-memory -convert-triton-amdgpu-to-llvm=arch=gfx1100 | FileCheck %s --check-prefix=LLVM

// FlagMega-on-Radeon pipe <-> warp_specialize integration: a pipe whose producer
// lives in a 1-warp partition and whose consumer is the 4-warp default group.
// Each barrier is sized by its endpoint's partition, not the whole CTA:
//   full  (writer -> reader) = producer threads  = 1 warp * 32 = 32
//   empty (reader -> writer) = consumer threads  = 4 warps * 32 = 128, init_phase 1
// The pipe identity/fields cross the warp-spec boundary as captures, so the key
// must resolve them to the same pipe (writer uses the partition block-args).

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // PIPE-LABEL: @pipe_in_warpspec
  // full sized by the 1-warp producer partition:
  // PIPE: amdg.init_barrier %{{.*}}, 32 :
  // empty sized by the 4-warp consumer, started free:
  // PIPE: amdg.init_barrier %{{.*}}, 128 {init_phase = 1 : i32}
  // PIPE-NOT: tle.pipe

  // WS-LABEL: @pipe_in_warpspec
  // WS-NOT: tle.pipe
  // WS-NOT: ttg.warp_specialize

  // LLVM-LABEL: llvm.func @pipe_in_warpspec
  // LLVM-NOT: tle.pipe
  // LLVM-NOT: ttg.warp_specialize
  tt.func public @pipe_in_warpspec(%fields: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %id = tle.pipe.create %fields {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    ttg.warp_specialize(%id, %fields) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      %closed = tle.pipe.reader_wait %id, %fields[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %id, %fields[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%idp: i32, %fp: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) num_warps(1) {
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      tle.pipe.writer_acquire %idp, %fp[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %idp, %fp[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (i32, !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
