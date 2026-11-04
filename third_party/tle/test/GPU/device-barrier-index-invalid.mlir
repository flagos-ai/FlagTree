// RUN: triton-opt %s -split-input-file -verify-diagnostics

module {
  tt.func @missing_capacity(%comm: i64, %slot: i32) {
    // expected-error @+1 {{dynamic_index requires a positive barrier_count}}
    "tle.device intra barrier"(%comm, %slot) <{barrier_type = "sync", coop_kind = 2 : i32, order = 3 : i32}> : (i64, i32) -> ()
    tt.return
  }
}

// -----

module {
  tt.func @ambiguous_index(%comm: i64, %slot: i32) {
    // expected-error @+1 {{requires exactly one of index or dynamic_index}}
    "tle.device intra barrier"(%comm, %slot) <{barrier_type = "sync", coop_kind = 2 : i32, index = 0 : i32, order = 3 : i32, barrier_count = 17 : i32}> : (i64, i32) -> ()
    tt.return
  }
}

// -----

module {
  tt.func @zero_capacity(%comm: i64, %slot: i32) {
    // expected-error @+1 {{barrier_count must be positive}}
    "tle.device intra barrier"(%comm, %slot) <{barrier_type = "sync", coop_kind = 2 : i32, order = 3 : i32, barrier_count = 0 : i32}> : (i64, i32) -> ()
    tt.return
  }
}

// -----

module {
  tt.func @constant_outside_capacity(%comm: i64) {
    %slot = arith.constant 17 : i32
    // expected-error @+1 {{constant barrier index is outside the declared capacity}}
    "tle.device intra barrier"(%comm, %slot) <{barrier_type = "sync", coop_kind = 2 : i32, order = 3 : i32, barrier_count = 17 : i32}> : (i64, i32) -> ()
    tt.return
  }
}
