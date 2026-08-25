// RUN: triton-opt %s -test-print-alignment 2>/dev/null | FileCheck %s

module {
  // CHECK: tt.func private @callee(%arg0: !tt.ptr<bf16> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64})
  tt.func private @callee(%arg0: !tt.ptr<bf16>) {
    %0 = tt.load %arg0 : !tt.ptr<bf16>
    tt.return
  }

  // CHECK: tt.func private @offset_callee(%arg0: !tt.ptr<bf16> {tt.constancy = 1 : i64, tt.contiguity = 1 : i64, tt.divisibility = 16 : i64})
  tt.func private @offset_callee(%arg0: !tt.ptr<bf16>) {
    %0 = tt.load %arg0 : !tt.ptr<bf16>
    tt.return
  }

  tt.func public @kernel(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    ttg.warp_specialize(%arg0)
    default {
      tt.call @callee(%arg0) : (!tt.ptr<bf16>) -> ()
      ttg.warp_yield
    }
    partition0(%arg1: !tt.ptr<bf16>) num_warps(4) {
      tt.call @callee(%arg1) : (!tt.ptr<bf16>) -> ()
      ttg.warp_return
    } : (!tt.ptr<bf16>) -> ()
    tt.return
  }


  tt.func public @offset_kernel(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: i32) {
    %c16_i32 = arith.constant 16 : i32
    %offset = arith.muli %arg1, %c16_i32 : i32
    %ptr_i8 = tt.addptr %arg0, %offset : !tt.ptr<i8>, i32
    %ptr = tt.bitcast %ptr_i8 : !tt.ptr<i8> -> !tt.ptr<bf16>
    ttg.warp_specialize(%ptr)
    default {
      tt.call @offset_callee(%ptr) : (!tt.ptr<bf16>) -> ()
      ttg.warp_yield
    }
    partition0(%arg2: !tt.ptr<bf16>) num_warps(4) {
      tt.call @offset_callee(%arg2) : (!tt.ptr<bf16>) -> ()
      ttg.warp_return
    } : (!tt.ptr<bf16>) -> ()
    tt.return
  }
}
