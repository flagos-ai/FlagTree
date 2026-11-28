// RUN: triton-opt %s -triton-tle-lower-pipe-to-rdna -triton-tle-restore-pipe-function-calls | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @entry
  tt.func public @entry(%out: !tt.ptr<i32>, %value: i32) {
    %id = tle.pipe.create {capacity = 1 : i32, field_names = [], scope = "cta", one_shot = true}
    // CHECK: %[[RESULT:.*]] = tt.call @sender{{[^ (]*}}({{.*}}) : ({{.*}}) -> i32
    %result = tt.call @sender(%id, %value) : (i32, i32) -> i32
    // CHECK: tt.store %{{.*}}, %[[RESULT]] : !tt.ptr<i32>
    tt.store %out, %result : !tt.ptr<i32>
    tt.return
  }

  // CHECK-LABEL: tt.func private @sender
  // CHECK-SAME: -> i32
  tt.func private @sender(%id: i32, %value: i32) -> i32 attributes {noinline = true} {
    %c0 = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    tle.pipe.writer_commit %id, [%c0] {capacity = 1 : i32, field_names = [], scope = "cta"}
    %result = arith.addi %value, %one : i32
    // CHECK: tt.return %{{.*}} : i32
    tt.return %result : i32
  }
}
