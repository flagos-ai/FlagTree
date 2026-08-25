// RUN: triton-opt %s -split-input-file -verify-diagnostics

module {
  tt.func @mismatched_arrays() {
    // expected-error@+1 {{expects offsets and counts to have equal length}}
    tle.init_barrier_group {counts = array<i32: 1>, offsets = array<i32: 0, 8>, worker_count = 32 : i32}
    tt.return
  }
}

// -----

module {
  tt.func @misaligned_offset() {
    // expected-error@+1 {{expects non-negative, 8-byte-aligned offsets; got 4}}
    tle.init_barrier_group {counts = array<i32: 1>, offsets = array<i32: 4>, worker_count = 32 : i32}
    tt.return
  }
}

// -----

module {
  tt.func @duplicate_offset() {
    // expected-error@+1 {{contains duplicate offset 0}}
    tle.init_barrier_group {counts = array<i32: 1, 1>, offsets = array<i32: 0, 0>, worker_count = 32 : i32}
    tt.return
  }
}

// -----

module {
  tt.func @invalid_worker_count() {
    // expected-error@+1 {{expects worker_count to be positive}}
    tle.init_barrier_group {counts = array<i32: 1>, offsets = array<i32: 0>, worker_count = 0 : i32}
    tt.return
  }
}
