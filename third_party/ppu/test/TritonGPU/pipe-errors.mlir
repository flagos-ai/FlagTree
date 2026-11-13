// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: triton-opt %s -split-input-file --tritonppu-tle-lower-pipe -verify-diagnostics

// Forms the PPU pipe lowering rejects before rewriting. Each diagnostic names
// what the caller wrote, so a pipe that cannot lower is not left to fail
// later in the conversion.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

// A lifecycle op whose pipe has no create in the same function: PPU pipes must
// be created and used in one kernel, after inlining.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @missing_create(%ring: !ttg.memdesc<4x128xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %phase = arith.constant false
    // expected-error @below {{has no matching pipe.create in this function}}
    tle.pipe.writer_acquire %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A reader naming a reader the pipe does not declare.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @undeclared_reader(%ring: !ttg.memdesc<4x128xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %phase = arith.constant false
    tle.pipe.create %ring {capacity = 4 : i32, field_names = ["data"], readers = ["local"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    // expected-error @below {{names reader 'remote' that the pipe does not declare}}
    %closed = tle.pipe.reader_wait %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], reader_name = "remote", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A reader naming a reader on a pipe that declares none.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @named_reader_without_declaration(%ring: !ttg.memdesc<4x128xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %phase = arith.constant false
    tle.pipe.create %ring {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    // expected-error @below {{names reader 'local' of a pipe without declared readers}}
    %closed = tle.pipe.reader_wait %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], reader_name = "local", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A pipe with a writer but no reader: the lowering needs both endpoints.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @no_reader(%ring: !ttg.memdesc<4x128xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %phase = arith.constant false
    // expected-error @below {{needs a writer and at least one reader in this kernel}}
    tle.pipe.create %ring {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.writer_acquire %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.writer_commit %ring[%c0] {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// Two readers of one pipe placed in the same warp group: a hand-over barrier
// counts the writer and one reader group, so each reader needs its own group.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @two_readers_one_group(%ring: !ttg.memdesc<4x128xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %phase = arith.constant false
    // expected-error @below {{run in the same warp group; one reader per group}}
    tle.pipe.create %ring {capacity = 4 : i32, field_names = ["data"], readers = ["a", "b"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.writer_acquire %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.writer_commit %ring[%c0] {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    %closed_a = tle.pipe.reader_wait %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], reader_name = "a", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.reader_release %ring[%c0] {capacity = 4 : i32, field_names = ["data"], reader_name = "a", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    %closed_b = tle.pipe.reader_wait %ring[%c0, %phase] {capacity = 4 : i32, field_names = ["data"], reader_name = "b", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tle.pipe.reader_release %ring[%c0] {capacity = 4 : i32, field_names = ["data"], reader_name = "b", scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// close on a one_shot pipe: a one_shot pipe has no close, as on NVIDIA.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @close_on_one_shot(%ring: !ttg.memdesc<1x128xf32, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %phase = arith.constant false
    tle.pipe.create %ring {capacity = 1 : i32, field_names = ["data"], one_shot = true, scope = "cta"} : !ttg.memdesc<1x128xf32, #shared, #smem, mutable>
    // expected-error @below {{does not support close on one_shot pipe}}
    tle.pipe.writer_close %ring[%c0, %phase] {capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A pipe handing over between warp groups inside a loop: the readers' initial
// arrivals and the hardware barrier state would carry over to the next
// execution of the warp_specialize.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @warp_specialize_in_loop(%ring: !ttg.memdesc<2x128xf32, #shared, #smem, mutable>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %phase = arith.constant false
    // expected-error @below {{hands over between warp groups inside a loop}}
    tle.pipe.create %ring {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
    scf.for %i = %c0 to %n step %c1 : i32 {
      ttg.warp_specialize(%ring)
      default {
        tle.pipe.writer_acquire %ring[%c0, %phase] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
        tle.pipe.writer_commit %ring[%c0] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
        ttg.warp_yield
      }
      partition0(%reader: !ttg.memdesc<2x128xf32, #shared, #smem, mutable>) num_warps(4) {
        %stage = arith.constant 0 : i32
        %parity = arith.constant false
        %closed = tle.pipe.reader_wait %reader[%stage, %parity] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
        tle.pipe.reader_release %reader[%stage] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
        ttg.warp_return
      } : (!ttg.memdesc<2x128xf32, #shared, #smem, mutable>) -> ()
    }
    tt.return
  }
}

// -----

// A one_shot pipe handing over between warp groups inside a loop: its
// counters are zero filled once, and nothing orders a second execution of the
// writer after the readers of the first.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @one_shot_warp_specialize_in_loop(%buf: !ttg.memdesc<1x128xf32, #shared, #smem, mutable>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %phase = arith.constant false
    // expected-error @below {{hands over between warp groups inside a loop; the hardware barriers and the shared-memory counters keep their state across executions}}
    tle.pipe.create %buf {capacity = 1 : i32, field_names = ["data"], one_shot = true, scope = "cta"} : !ttg.memdesc<1x128xf32, #shared, #smem, mutable>
    scf.for %i = %c0 to %n step %c1 : i32 {
      ttg.warp_specialize(%buf)
      default {
        tle.pipe.writer_acquire %buf[%c0, %phase] {capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x128xf32, #shared, #smem, mutable>
        tle.pipe.writer_commit %buf[%c0] {capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x128xf32, #shared, #smem, mutable>
        ttg.warp_yield
      }
      partition0(%reader: !ttg.memdesc<1x128xf32, #shared, #smem, mutable>) num_warps(4) {
        %stage = arith.constant 0 : i32
        %parity = arith.constant false
        %closed = tle.pipe.reader_wait %reader[%stage, %parity] {capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x128xf32, #shared, #smem, mutable>
        tle.pipe.reader_release %reader[%stage] {capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x128xf32, #shared, #smem, mutable>
        ttg.warp_return
      } : (!ttg.memdesc<1x128xf32, #shared, #smem, mutable>) -> ()
    }
    tt.return
  }
}

// -----

// A pipe whose writer and reader sit in the partitions of two different
// warp_specialize ops: those partitions never run at the same time, so the
// hand-over barriers would count threads that never arrive.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @endpoints_in_two_warp_specializes(%ring: !ttg.memdesc<2x128xf32, #shared, #smem, mutable>) {
    // expected-error @below {{hands over to reader '' between warp groups that do not run at the same time}}
    tle.pipe.create %ring {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
    // expected-note @below {{the writer runs here}}
    ttg.warp_specialize(%ring)
    default {
      ttg.warp_yield
    }
    partition0(%writer: !ttg.memdesc<2x128xf32, #shared, #smem, mutable>) num_warps(4) {
      %stage = arith.constant 0 : i32
      %parity = arith.constant false
      tle.pipe.writer_acquire %writer[%stage, %parity] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
      tle.pipe.writer_commit %writer[%stage] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x128xf32, #shared, #smem, mutable>) -> ()
    // expected-note @below {{this reader runs here}}
    ttg.warp_specialize(%ring)
    default {
      ttg.warp_yield
    }
    partition0(%reader: !ttg.memdesc<2x128xf32, #shared, #smem, mutable>) num_warps(4) {
      %stage = arith.constant 0 : i32
      %parity = arith.constant false
      %closed = tle.pipe.reader_wait %reader[%stage, %parity] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
      tle.pipe.reader_release %reader[%stage] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x128xf32, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

// Two creates of the same capacity, scope, name and fields are one pipe. Pipes
// on one allocation differ by name or by the stages they ring through.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @duplicate_create(%ring: !ttg.memdesc<4x128xf32, #shared, #smem, mutable>) {
    // expected-note @below {{earlier pipe.create}}
    tle.pipe.create %ring {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    // expected-error @below {{duplicates an earlier pipe.create of the same capacity, scope, name and fields}}
    tle.pipe.create %ring {capacity = 4 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<4x128xf32, #shared, #smem, mutable>
    tt.return
  }
}

// -----

// A cross-group pipe whose writer runs outside the warp_specialize that holds
// its reader. The default warps block on the empty barrier before they enter
// the op, and the arrivals that would release it sit in a partition they have
// not started yet.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @writer_outside_warp_specialize(%ring: !ttg.memdesc<2x128xf32, #shared, #smem, mutable>) {
    %stage = arith.constant 0 : i32
    %parity = arith.constant false
    // expected-error @below {{hands over to reader '' between warp groups that do not run at the same time}}
    tle.pipe.create %ring {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
    // expected-note @below {{the writer runs here}}
    tle.pipe.writer_acquire %ring[%stage, %parity] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
    tle.pipe.writer_commit %ring[%stage] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
    // expected-note @below {{this reader runs here}}
    ttg.warp_specialize(%ring)
    default {
      ttg.warp_yield
    }
    partition0(%reader: !ttg.memdesc<2x128xf32, #shared, #smem, mutable>) num_warps(4) {
      %rstage = arith.constant 0 : i32
      %rparity = arith.constant false
      %closed = tle.pipe.reader_wait %reader[%rstage, %rparity] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
      tle.pipe.reader_release %reader[%rstage] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x128xf32, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----

// A close inside a loop the pipe is created outside of. The close tags are
// zero filled once, at the create, and nothing clears them, so the second
// execution would start with the stage already tagged and its readers would
// see the pipe closed before their first wait.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @close_inside_a_loop(%ring: !ttg.memdesc<2x128xf32, #shared, #smem, mutable>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %parity = arith.constant false
    // expected-note @below {{pipe created here}}
    tle.pipe.create %ring {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
    ttg.warp_specialize(%ring, %n)
    default {
      scf.for %i = %c0 to %n step %c1 : i32 {
        tle.pipe.writer_acquire %ring[%c0, %parity] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
        tle.pipe.writer_commit %ring[%c0] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
        // expected-error @below {{closes the pipe inside a loop the pipe is created outside of}}
        tle.pipe.writer_close %ring[%c1, %parity] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
      }
      ttg.warp_yield
    }
    partition0(%reader: !ttg.memdesc<2x128xf32, #shared, #smem, mutable>, %rn: i32) num_warps(4) {
      %rc0 = arith.constant 0 : i32
      %rc1 = arith.constant 1 : i32
      %rparity = arith.constant false
      scf.for %i = %rc0 to %rn step %rc1 : i32 {
        %closed = tle.pipe.reader_wait %reader[%rc0, %rparity] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
        tle.pipe.reader_release %reader[%rc0] {capacity = 2 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<2x128xf32, #shared, #smem, mutable>
      }
      ttg.warp_return
    } : (!ttg.memdesc<2x128xf32, #shared, #smem, mutable>, i32) -> ()
    tt.return
  }
}
