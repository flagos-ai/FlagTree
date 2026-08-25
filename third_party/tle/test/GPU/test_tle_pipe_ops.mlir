// RUN: triton-opt %s -split-input-file -verify-diagnostics | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @valid_pipe_ops
  tt.func @valid_pipe_ops(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    // CHECK: tle.pipe.create
    %pipe_identity_38 = tle.pipe.create %a {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.writer_acquire
    tle.pipe.writer_acquire %pipe_identity_38, %a[%c0, %false] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.writer_commit
    tle.pipe.writer_commit %pipe_identity_38, %a[%c0] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.writer_close
    tle.pipe.writer_close %pipe_identity_38, %a[%c0, %false] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.reader_wait
    %closed = tle.pipe.reader_wait %pipe_identity_38, %a[%c0, %false] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.reader_release
    tle.pipe.reader_release %pipe_identity_38, %a[%c0] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.drain
    tle.pipe.drain %pipe_identity_38, %a {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
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
    %pipe_identity_37 = tle.pipe.create %a {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], readers = ["left", "right"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.reader_wait
    // CHECK-SAME: reader_name = "left"
    %closed = tle.pipe.reader_wait %pipe_identity_37, %a[%c0, %false] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    // CHECK: tle.pipe.reader_release
    // CHECK-SAME: reader_name = "left"
    tle.pipe.reader_release %pipe_identity_37, %a[%c0] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], reader_name = "left", scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_non_cta_scope(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{MVP supports only scope = "cta"}}
    %pipe_identity_36 = tle.pipe.create %a {capacity = 4 : i32, field_names = ["a"], scope = "device"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_duplicate_reader_name(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{expects unique pipe reader names}}
    %pipe_identity_35 = tle.pipe.create %a {capacity = 4 : i32, field_names = ["a"], readers = ["left", "left"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_empty_readers(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{expects reader to contain at least one name}}
    %pipe_identity_34 = tle.pipe.create %a {capacity = 4 : i32, field_names = ["a"], readers = [], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
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
    %missing_pipe_identity_33 = arith.constant 0 : i32
    // expected-error @+1 {{expects valid public pipe reader_name}}
    %closed = tle.pipe.reader_wait %missing_pipe_identity_33, %a[%c0, %false] {capacity = 4 : i32, field_names = ["a"], reader_name = "readers", scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_capacity_mismatch(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{expects field leading dimension to equal pipe capacity}}
    %pipe_identity_32 = tle.pipe.create %a {capacity = 4 : i32, field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared1d = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_rank_one_field(%a: !ttg.memdesc<4xf16, #shared1d, #smem, mutable>) {
    // expected-error @+1 {{expects pipe fields to have rank >= 2}}
    %pipe_identity_31 = tle.pipe.create %a {capacity = 4 : i32, field_names = ["a"], scope = "cta"} : !ttg.memdesc<4xf16, #shared1d, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @reject_reserved_field_name(%a: !ttg.memdesc<4x16xf16, #shared, #smem, mutable>) {
    // expected-error @+1 {{expects valid public pipe field names}}
    %pipe_identity_30 = tle.pipe.create %a {capacity = 4 : i32, field_names = ["fields"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
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
    %missing_pipe_identity_29 = arith.constant 0 : i32
    %closed = tle.pipe.reader_wait %missing_pipe_identity_29, %a[%c0, %false] {capacity = 4 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<4x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tt.return
  }
}
