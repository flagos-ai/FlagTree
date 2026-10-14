// RUN: triton-opt %s -triton-tle-lower-pipe-to-rdna | FileCheck %s --check-prefix=PIPE

// FlagMega-on-Radeon multi-reader pipe: one writer partition feeds TWO distinct
// reader endpoints (the 4-warp default group AND a 1-warp partition). The empty
// barrier (reader -> writer) must be sized by the SUM of the distinct readers'
// thread counts, not by whichever reader the pre-scan happened to visit last:
//   full  (writer -> reader) = producer threads = 1 warp  * 32 = 32
//   empty (readers -> writer) = 4-warp default (128) + 1-warp partition (32) = 160
// A regression that overwrites instead of accumulating would size the empty
// barrier at 32 or 128 and under-count the readers, hanging the pipe.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // PIPE-LABEL: @pipe_multi_reader
  // full sized by the 1-warp producer partition:
  // PIPE: amdg.init_barrier %{{.*}}, 32 :
  // empty sized by the SUM of the two distinct readers (128 + 32), started free:
  // PIPE: amdg.init_barrier %{{.*}}, 160 {init_phase = 1 : i32}
  // PIPE-NOT: tle.pipe
  tt.func public @pipe_multi_reader(%fields: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %id = tle.pipe.create %fields {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    ttg.warp_specialize(%id, %fields) attributes {warpGroupStartIds = array<i32: 4, 5>}
    default {
      // Reader A: the 4-warp default group (128 threads).
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      %closed = tle.pipe.reader_wait %id, %fields[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %id, %fields[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%idp: i32, %fp: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) num_warps(1) {
      // Writer: the 1-warp producer partition (32 threads).
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      tle.pipe.writer_acquire %idp, %fp[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %idp, %fp[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    }
    partition1(%idr: i32, %fr: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) num_warps(1) {
      // Reader B: a second 1-warp reader partition (32 threads).
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      %closed2 = tle.pipe.reader_wait %idr, %fr[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %idr, %fr[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (i32, !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
