// RUN: triton-opt %s -split-input-file -triton-tle-lower-pipe-to-rdna | FileCheck %s --check-prefix=TTGIR
// RUN: triton-opt %s -split-input-file -triton-tle-lower-pipe-to-rdna -allocate-shared-memory -convert-triton-amdgpu-to-llvm=arch=gfx1100 | FileCheck %s --check-prefix=LLVM

// FlagMega-on-Radeon Phase 1 (P1-A): tle.pipe.* must lower to the AMD mbarrier
// ops (amdg.{init,arrive,wait}_barrier) with a per-stage full/empty pair, and no
// tle.pipe op may survive. The second run checks the whole chain reaches LLVM on
// gfx1100 (the arrive emulation becomes an llvm.cmpxchg on the i64 LDS word).

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // TTGIR-LABEL: @pipe_roundtrip
  // TTGIR: amdg.init_barrier
  // TTGIR: amdg.arrive_barrier
  // TTGIR: amdg.wait_barrier
  // TTGIR-NOT: tle.pipe

  // LLVM-LABEL: llvm.func @pipe_roundtrip
  // LLVM: llvm.cmpxchg
  // LLVM-NOT: tle.
  tt.func public @pipe_roundtrip() {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %fields = ttg.local_alloc : () -> !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %id = tle.pipe.create %fields {capacity = 2 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %id, %fields[%c0, %false] {capacity = 2 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %id, %fields[%c0] {capacity = 2 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_close %id, %fields[%c0, %false] {capacity = 2 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %closed = tle.pipe.reader_wait %id, %fields[%c0, %false] {capacity = 2 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.reader_release %id, %fields[%c0] {capacity = 2 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.drain %id, %fields {capacity = 2 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// One-shot pipe: no close-tag ring, is_closed folds to a constant false.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // TTGIR-LABEL: @pipe_one_shot
  // TTGIR: amdg.init_barrier
  // TTGIR-NOT: tle.pipe
  tt.func public @pipe_one_shot() {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %fields = ttg.local_alloc : () -> !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %id = tle.pipe.create %fields {capacity = 1 : i32, one_shot = true, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %id, %fields[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %id, %fields[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %closed = tle.pipe.reader_wait %id, %fields[%c0, %false] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.reader_release %id, %fields[%c0] {capacity = 1 : i32, pipe_name = "p", field_names = ["p"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}
