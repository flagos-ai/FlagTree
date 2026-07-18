// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @valid_pipe_ops
  tt.func @valid_pipe_ops(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // CHECK: tle.pipe.create
    tle.pipe.create %a {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.writer_acquire
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.writer_commit
    tle.pipe.writer_commit %a[%c0] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.writer_close
    tle.pipe.writer_close %a[%c0, %false] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.reader_wait
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.reader_release
    tle.pipe.reader_release %a[%c0] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @valid_multi_reader_pipe_ops
  tt.func @valid_multi_reader_pipe_ops(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // CHECK: tle.pipe.create
    // CHECK-SAME: readers = ["left", "right"]
    tle.pipe.create %a {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], readers = ["left", "right"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.reader_wait
    // CHECK-SAME: reader_name = "left"
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.reader_release
    // CHECK-SAME: reader_name = "left"
    tle.pipe.reader_release %a[%c0] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_non_cta_scope(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{MVP supports only scope = "cta"}}
    tle.pipe.create %a {capacity = 4 : i32, field_names = ["a"], scope = "device"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_duplicate_reader_name(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{expects unique pipe reader names}}
    tle.pipe.create %a {capacity = 4 : i32, field_names = ["a"], readers = ["left", "left"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_empty_readers(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{expects reader to contain at least one name}}
    tle.pipe.create %a {capacity = 4 : i32, field_names = ["a"], readers = [], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_reserved_reader_name(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // expected-error @+1 {{expects valid public pipe reader_name}}
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 4 : i32, field_names = ["a"], reader_name = "readers", scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_capacity_mismatch(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{expects field leading dimension to equal pipe capacity}}
    tle.pipe.create %a {capacity = 4 : i32, field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared1d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_rank_one_field(%a: !ttg.memdesc<4xf16, #shared1d, #smem, mutable>) {
    // expected-error @+1 {{expects pipe fields to have rank >= 2}}
    tle.pipe.create %a {capacity = 4 : i32, field_names = ["a"], scope = "cta"} : !ttg.memdesc<4xf16, #shared1d, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_reserved_field_name(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{expects valid public pipe field names}}
    tle.pipe.create %a {capacity = 4 : i32, field_names = ["fields"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @valid_close_result_use
  tt.func @valid_close_result_use(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // CHECK: tle.pipe.reader_wait
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tt.return
  }
}
