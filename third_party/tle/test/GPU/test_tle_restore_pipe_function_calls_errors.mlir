// RUN: not triton-opt %s --triton-tle-restore-pipe-function-calls 2>&1 | FileCheck %s

module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: error: 'tle.pipe.call_begin' op calls to pipe helper decode_role do not have one structurally equivalent lowered body
  tt.func public @kernel(%first: !tt.ptr<i32>, %second: !tt.ptr<i32>,
                         %offset: i32) {
    %first_arg, %first_offset = tle.pipe.call_begin %first, %offset {callee = "decode_role", call_id = 0 : i64} : (!tt.ptr<i32>, i32) -> (!tt.ptr<i32>, i32)
    %first_value = tt.load %first_arg : !tt.ptr<i32>
    %first_result = arith.addi %first_value, %first_offset : i32
    tt.store %first_arg, %first_result : !tt.ptr<i32>
    tle.pipe.call_end {callee = "decode_role", call_id = 0 : i64}

    %second_arg, %second_offset = tle.pipe.call_begin %second, %offset {callee = "decode_role", call_id = 1 : i64} : (!tt.ptr<i32>, i32) -> (!tt.ptr<i32>, i32)
    %second_value = tt.load %second_arg : !tt.ptr<i32>
    %second_result = arith.muli %second_value, %second_offset : i32
    tt.store %second_arg, %second_result : !tt.ptr<i32>
    tle.pipe.call_end {callee = "decode_role", call_id = 1 : i64}
    tt.return
  }
}
