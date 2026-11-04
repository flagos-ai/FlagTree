// RUN: triton-opt %s -triton-tle-lower-pipe-to-rdna | FileCheck %s --check-prefix=PIPE

// FlagMega-on-Radeon multi-reader pipe: the pipe DECLARES two readers ("a" and
// "b") and both run in the same 4-warp default group. Readers are identified by
// their logical name, not by the physical region they run in, so each distinct
// declared reader must arrive before a slot can be reused: the empty barrier is
// the SUM of the two named readers (128 + 128 = 256). Identifying readers by
// region instead would collapse the two same-group readers to a single 128.
//   full  (writer -> reader) = producer threads = 1 warp  * 32 = 32
//   empty (readers -> writer) = reader "a" (128) + reader "b" (128) = 256

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // PIPE-LABEL: @pipe_multi_named_reader
  // full sized by the 1-warp producer partition:
  // PIPE: amdg.init_barrier %{{.*}}, 32 :
  // empty sized by the SUM of the two distinct NAMED readers (128 + 128), free:
  // PIPE: amdg.init_barrier %{{.*}}, 256 {init_phase = 1 : i32}
  // PIPE-NOT: tle.pipe
  tt.func public @pipe_multi_named_reader(%fields: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %id = tle.pipe.create %fields {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta", readers = ["a", "b"]} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    ttg.warp_specialize(%id, %fields) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      // Reader "a": the 4-warp default group (128 threads).
      %ca = tle.pipe.reader_wait %id, %fields[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta", reader_name = "a"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %id, %fields[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta", reader_name = "a"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      // Reader "b": a second declared reader, same default group (128 threads).
      %cb = tle.pipe.reader_wait %id, %fields[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta", reader_name = "b"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.reader_release %id, %fields[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta", reader_name = "b"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    }
    partition0(%idp: i32, %fp: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) num_warps(1) {
      // Writer: the 1-warp producer partition (32 threads).
      %c0 = arith.constant 0 : i32
      %false = arith.constant false
      tle.pipe.writer_acquire %idp, %fp[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      tle.pipe.writer_commit %idp, %fp[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (i32, !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
