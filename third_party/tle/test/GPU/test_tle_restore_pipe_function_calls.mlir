// RUN: triton-opt %s --triton-tle-restore-pipe-function-calls | FileCheck %s

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func public @kernel
  // CHECK-NOT: tle.pipe.call_
  // CHECK: tt.call @decode_role
  // CHECK-NOT: tle.pipe.call_
  // CHECK: tt.call @decode_role
  // CHECK-NOT: tle.pipe.call_
  tt.func public @kernel(%first: !tt.ptr<i32>, %second: !tt.ptr<i32>,
                         %first_offset: i32, %second_offset: i32) {
    %first_arg, %first_offset_arg = tle.pipe.call_begin %first, %first_offset {callee = "decode_role", call_id = 0 : i64} : (!tt.ptr<i32>, i32) -> (!tt.ptr<i32>, i32)
    %first_value = tt.load %first_arg : !tt.ptr<i32>
    %first_result = arith.addi %first_value, %first_offset_arg : i32
    tt.store %first_arg, %first_result : !tt.ptr<i32>
    tle.pipe.call_end {callee = "decode_role", call_id = 0 : i64}

    %second_arg, %second_offset_arg = tle.pipe.call_begin %second, %second_offset {callee = "decode_role", call_id = 1 : i64} : (!tt.ptr<i32>, i32) -> (!tt.ptr<i32>, i32)
    %second_value = tt.load %second_arg : !tt.ptr<i32>
    %second_result = arith.addi %second_value, %second_offset_arg : i32
    tt.store %second_arg, %second_result : !tt.ptr<i32>
    tle.pipe.call_end {callee = "decode_role", call_id = 1 : i64}
    tt.return
  }

  // CHECK-LABEL: tt.func public @constant_kernel
  // CHECK: tt.call @constant_role
  // CHECK: tt.call @constant_role
  tt.func public @constant_kernel(%first: !tt.ptr<i32>, %second: !tt.ptr<i32>) {
    %first_one = arith.constant 1 : i32
    %second_one = arith.constant 1 : i32
    %first_arg = tle.pipe.call_begin %first {callee = "constant_role", call_id = 2 : i64} : (!tt.ptr<i32>) -> !tt.ptr<i32>
    %first_value = tt.load %first_arg : !tt.ptr<i32>
    %first_result = arith.addi %first_value, %first_one : i32
    tt.store %first_arg, %first_result : !tt.ptr<i32>
    tle.pipe.call_end {callee = "constant_role", call_id = 2 : i64}
    %second_arg = tle.pipe.call_begin %second {callee = "constant_role", call_id = 3 : i64} : (!tt.ptr<i32>) -> !tt.ptr<i32>
    %second_value = tt.load %second_arg : !tt.ptr<i32>
    %second_result = arith.addi %second_value, %second_one : i32
    tt.store %second_arg, %second_result : !tt.ptr<i32>
    tle.pipe.call_end {callee = "constant_role", call_id = 3 : i64}
    tt.return
  }

  // CHECK-LABEL: tt.func public @token_kernel
  // CHECK: tt.call @token_role
  // CHECK: tt.call @token_role
  tt.func public @token_kernel(%first: tensor<2x!nvws.token>,
                               %second: tensor<2x!nvws.token>,
                               %idx: i32, %phase: i1) {
    %first_token, %first_idx, %first_phase = tle.pipe.call_begin %first, %idx, %phase {callee = "token_role", call_id = 4 : i64} : (tensor<2x!nvws.token>, i32, i1) -> (tensor<2x!nvws.token>, i32, i1)
    nvws.consumer_wait %first, %first_idx, %first_phase {async_task_id = array<i32: 0>} : tensor<2x!nvws.token>, i32, i1
    tle.pipe.call_end {callee = "token_role", call_id = 4 : i64}
    %second_token, %second_idx, %second_phase = tle.pipe.call_begin %second, %idx, %phase {callee = "token_role", call_id = 5 : i64} : (tensor<2x!nvws.token>, i32, i1) -> (tensor<2x!nvws.token>, i32, i1)
    nvws.consumer_wait %second, %second_idx, %second_phase {async_task_id = array<i32: 0>} : tensor<2x!nvws.token>, i32, i1
    tle.pipe.call_end {callee = "token_role", call_id = 5 : i64}
    tt.return
  }

  // CHECK-LABEL: tt.func private @decode_role(
  // CHECK-SAME: !tt.ptr<i32>
  // CHECK-SAME: i32
  // CHECK-SAME: attributes {noinline = true, "ttg.num-warps" = 8 : i32}
  // CHECK: tt.load
  // CHECK: arith.addi
  // CHECK: tt.store
  // CHECK: tt.return

  // CHECK-LABEL: tt.func private @constant_role(
  // CHECK-SAME: !tt.ptr<i32>
  // CHECK-SAME: ) attributes
  // CHECK: arith.constant 1 : i32
  // CHECK: arith.addi

  // CHECK-LABEL: tt.func private @token_role(
  // CHECK: nvws.consumer_wait
}
