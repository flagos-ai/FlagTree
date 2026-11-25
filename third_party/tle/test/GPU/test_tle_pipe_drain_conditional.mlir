// RUN: triton-opt %s -split-input-file --triton-tle-lower-pipe-to-nvws -verify-diagnostics
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @conditional_writer_drain(%field: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>, %condition: i1) {
    %pipe = tle.pipe.create %field {capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.writer_close %pipe, %field[%zero, %false] {async_task_id = array<i32: 0>, capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    scf.if %condition {
      // expected-error @+1 {{wait_drained must execute once unconditionally in its async task}}
      tle.pipe.drain %pipe, %field {async_task_id = array<i32: 0>, capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    }
    tle.pipe.reader_release %pipe, %field[%zero] {async_task_id = array<i32: 1>, capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.drain %pipe, %field {async_task_id = array<i32: 1>, capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @loop_writer_drain(%field: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>, %condition: i1) {
    %pipe = tle.pipe.create %field {capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.writer_close %pipe, %field[%zero, %false] {async_task_id = array<i32: 0>, capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %lb = arith.constant 0 : index
    %ub = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %iv = %lb to %ub step %step {
      // expected-error @+1 {{wait_drained must execute once unconditionally in its async task}}
      tle.pipe.drain %pipe, %field {async_task_id = array<i32: 0>, capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    }
    tle.pipe.reader_release %pipe, %field[%zero] {async_task_id = array<i32: 1>, capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.drain %pipe, %field {async_task_id = array<i32: 1>, capacity = 1 : i32, field_names = ["data"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}
