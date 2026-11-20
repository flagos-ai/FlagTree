// RUN: triton-opt %s -triton-tle-lower-pipe-to-nvws | FileCheck %s

// CHECK-DAG: #[[$ONE:.*]] = #ttg.blocked<{{.*}}warpsPerCTA = [1]{{.*}}>
// CHECK-DAG: #[[$EIGHT:.*]] = #ttg.blocked<{{.*}}warpsPerCTA = [8]{{.*}}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @partition_writer_close
  tt.func @partition_writer_close(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    ttg.warp_specialize(%a) default {
      // CHECK: ttg.local_load {{.*}} -> tensor<1xi32, #[[$EIGHT]]>
      %closed = tle.pipe.reader_wait %a[%zero, %false] {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      scf.if %closed {}
      tle.pipe.reader_release %a[%zero] {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    } partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) num_warps(1) {
      %part_zero = arith.constant 0 : i32
      %part_false = arith.constant false
      // CHECK: partition0
      // CHECK: tt.splat {{.*}} -> tensor<1xi32, #[[$ONE]]>
      tle.pipe.writer_close %arg0[%part_zero, %part_false] {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>) -> ()
    tt.return
  }

  // CHECK-LABEL: tt.func @partition_reader_close
  tt.func @partition_reader_close(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %zero = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    ttg.warp_specialize(%a) default {
      // CHECK: tt.splat {{.*}} -> tensor<1xi32, #[[$EIGHT]]>
      tle.pipe.writer_close %a[%zero, %false] {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_yield
    } partition0(%arg0: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) num_warps(1) {
      %part_zero = arith.constant 0 : i32
      %part_false = arith.constant false
      // CHECK: partition0
      // CHECK: ttg.local_load {{.*}} -> tensor<1xi32, #[[$ONE]]>
      %closed = tle.pipe.reader_wait %arg0[%part_zero, %part_false] {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      scf.if %closed {}
      tle.pipe.reader_release %arg0[%part_zero] {capacity = 2 : i32, pipe_name = "close", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<2x16xf16, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
