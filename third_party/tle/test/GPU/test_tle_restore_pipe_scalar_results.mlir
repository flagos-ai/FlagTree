// RUN: triton-opt %s --triton-tle-restore-pipe-function-calls | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @scalar_results
  tt.func public @scalar_results(%out: !tt.ptr<i32>, %a: i32, %b: i32) {
    %aa = tle.pipe.call_begin %a {callee = "plus_one", call_id = 0 : i64} : (i32) -> i32
    %one = arith.constant 1 : i32
    %sum_a = arith.addi %aa, %one : i32
    tle.pipe.call_end {callee = "plus_one", call_id = 0 : i64}
    %ra = tle.pipe.call_return %sum_a {call_id = 0 : i64} : (i32) -> i32
    // CHECK: %[[A:.+]] = tt.call @plus_one
    tt.store %out, %ra : !tt.ptr<i32>
    %bb = tle.pipe.call_begin %b {callee = "plus_one", call_id = 1 : i64} : (i32) -> i32
    %one_b = arith.constant 1 : i32
    %sum_b = arith.addi %bb, %one_b : i32
    tle.pipe.call_end {callee = "plus_one", call_id = 1 : i64}
    %rb = tle.pipe.call_return %sum_b {call_id = 1 : i64} : (i32) -> i32
    // CHECK: %[[B:.+]] = tt.call @plus_one
    tt.store %out, %rb : !tt.ptr<i32>
    tt.return
  }
  // A return that aliases an original argument also remains a result.
  // CHECK-LABEL: tt.func public @direct_result
  tt.func public @direct_result(%out: !tt.ptr<i32>, %value: i32) {
    %alias = tle.pipe.call_begin %value {callee = "identity", call_id = 2 : i64} : (i32) -> i32
    tle.pipe.call_end {callee = "identity", call_id = 2 : i64}
    %r = tle.pipe.call_return %alias {call_id = 2 : i64} : (i32) -> i32
    // CHECK: tt.call @identity
    tt.store %out, %r : !tt.ptr<i32>
    tt.return
  }
  // CHECK-LABEL: tt.func private @plus_one
  // CHECK-SAME: -> i32
  // CHECK: arith.addi
  // CHECK: tt.return %{{.*}} : i32
  // CHECK-LABEL: tt.func private @identity
  // CHECK-SAME: -> i32
  // CHECK: tt.return %{{.*}} : i32
}
