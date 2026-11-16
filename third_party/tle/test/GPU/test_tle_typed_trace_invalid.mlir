// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -split-input-file -verify-diagnostics

module {
  llvm.func @bad_kind(%layer: i32, %iter: i32, %slot: i32, %phase: i32) {
    // expected-error @+1 {{typed trace event kind must be a supported pipe or module event}}
    "tle.trace_event"(%layer, %iter, %slot, %phase) {kind = 999 : i32, site = 1 : i32} : (i32, i32, i32, i32) -> ()
    llvm.return
  }
}

// -----

module {
  llvm.func @bad_site(%layer: i32, %iter: i32, %slot: i32, %phase: i32) {
    // expected-error @+1 {{typed trace site must be nonnegative}}
    "tle.trace_event"(%layer, %iter, %slot, %phase) {kind = 256 : i32, site = -1 : i32} : (i32, i32, i32, i32) -> ()
    llvm.return
  }
}

// -----

module attributes {ttg.target = "hip:gfx942"} {
  llvm.func @unsupported(%layer: i32, %iter: i32, %slot: i32, %phase: i32) {
    // expected-error @+1 {{typed trace requires the NVIDIA warp-specialize backend}}
    "tle.trace_event"(%layer, %iter, %slot, %phase) {kind = 256 : i32, site = 1 : i32} : (i32, i32, i32, i32) -> ()
    llvm.return
  }
}

// -----

module {
  tt.func @identity_misused_as_stage(%id: i32, %layer: i32, %iter: i32, %phase: i1) {
    // expected-error @+1 {{pipe trace context may only annotate acquire/wait/release identities}}
    %trace = "tle.pipe_trace_context"(%id, %layer, %iter) {site = 1 : i32} : (i32, i32, i32) -> i32
    "tle.pipe.writer_acquire"(%id, %trace, %phase) {capacity = 2 : i32, scope = "cta", field_names = []} : (i32, i32, i1) -> ()
    tt.return
  }
}
