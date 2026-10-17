// RUN: triton-opt %s -split-input-file -triton-tle-lower-pipe-to-rdna | FileCheck %s

// FlagMega-on-Radeon P1-A: behavior coverage for the RDNA pipe lowering, the
// AMD counterpart of the NVWS pipe-behavior suite. The RDNA lowering is
// correctness-first (whole-CTA participants, no NVWS participant-count
// inference), so these assert what the mbarrier lowering actually emits.

// Non-power-of-two capacity: NVWS pads the control token to a power of two, but
// the RDNA barrier arrays index the logical capacity directly, so C=3 stays a
// 3-stage array (not padded to 4) with 3 full+empty init pairs.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipe_capacity_three
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<3x1xi64
  // CHECK-COUNT-6: amdg.init_barrier
  // CHECK-NOT: tle.pipe
  tt.func public @pipe_capacity_three() {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %fields = ttg.local_alloc : () -> !ttg.memdesc<3x16xf16, #shared, #smem, mutable>
    %id = tle.pipe.create %fields {capacity = 3 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<3x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %id, %fields[%c0, %false] {capacity = 3 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<3x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %id, %fields[%c0] {capacity = 3 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<3x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Whole-CTA participant count: init uses num_warps * threads_per_warp (no
// participant inference), per-thread arrive uses count 1. Here 8 warps * 32 =
// 256 (the C=2 roundtrip test already covers the 4-warp = 128 case).

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipe_counts_8warps
  // CHECK: amdg.init_barrier %{{.*}}, 256
  // CHECK: amdg.arrive_barrier %{{.*}}, 1
  // CHECK-NOT: tle.pipe
  tt.func public @pipe_counts_8warps() {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %fields = ttg.local_alloc : () -> !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %id = tle.pipe.create %fields {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %id, %fields[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %id, %fields[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Close on a non-zero stage: writer_close writes the close tag into the close
// ring slot for that stage, then publishes on that stage's full mbarrier.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipe_close_stage1
  // close tag ring is allocated (non-one-shot) and written by writer_close
  // CHECK: ttg.local_alloc %{{.*}} : (tensor<2x1xi32{{.*}}>) -> !ttg.memdesc<2x1xi32
  // CHECK: ttg.local_store
  // CHECK-NOT: tle.pipe
  tt.func public @pipe_close_stage1() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %false = arith.constant false
    %fields = ttg.local_alloc : () -> !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %id = tle.pipe.create %fields {capacity = 2 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %id, %fields[%c0, %false] {capacity = 2 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %id, %fields[%c0] {capacity = 2 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_close %id, %fields[%c1, %false] {capacity = 2 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A pipe created with an explicit reader name lowers correctly: the RDNA path
// treats the reader as the whole CTA (reader_name is accepted and threaded, but
// per-reader task counts are a warp-specialization concern, P1-B).

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipe_named_reader
  // CHECK: amdg.wait_barrier
  // CHECK: amdg.arrive_barrier
  // CHECK-NOT: tle.pipe
  tt.func public @pipe_named_reader() {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %fields = ttg.local_alloc : () -> !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %id = tle.pipe.create %fields {capacity = 1 : i32, one_shot = true, pipe_name = "p", field_names = ["p"], readers = ["worker"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %id, %fields[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %id, %fields[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %closed = tle.pipe.reader_wait %id, %fields[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], reader_name = "worker", scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.reader_release %id, %fields[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], reader_name = "worker", scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}
