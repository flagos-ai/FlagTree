// RUN: triton-opt --triton-tle-lower-pipe-to-nvws %s | FileCheck %s

// Pipe capacity is a logical ring size, not a Triton tensor extent. Close-tag
// and token storage must be padded without changing data-ring indices/phases.
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @capacity_three
  // CHECK: tt.splat {{.*}} : i32 -> tensor<4x1xi32
  // CHECK: ttg.local_alloc {{.*}} -> !ttg.memdesc<4x1xi32
  // CHECK: nvws.create_token {{.*}}numBuffers = 4 : i32
  tt.func @capacity_three(%a: !ttg.memdesc<3x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    %pipe = tle.pipe.create %a {capacity = 3 : i32, pipe_name = "three", field_names = ["a"], scope = "cta"} : !ttg.memdesc<3x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %pipe, %a[%c0, %false] {capacity = 3 : i32, pipe_name = "three", field_names = ["a"], scope = "cta"} : !ttg.memdesc<3x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %pipe, %a[%c0] {capacity = 3 : i32, pipe_name = "three", field_names = ["a"], scope = "cta"} : !ttg.memdesc<3x16xf16, #shared, #smem, mutable>
    %closed = tle.pipe.reader_wait %pipe, %a[%c0, %false] {capacity = 3 : i32, pipe_name = "three", field_names = ["a"], scope = "cta"} : !ttg.memdesc<3x16xf16, #shared, #smem, mutable>
    tle.pipe.reader_release %pipe, %a[%c0] {capacity = 3 : i32, pipe_name = "three", field_names = ["a"], scope = "cta"} : !ttg.memdesc<3x16xf16, #shared, #smem, mutable>
    tt.return
  }
}
