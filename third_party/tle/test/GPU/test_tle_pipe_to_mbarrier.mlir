// RUN: triton-opt --triton-tle-lower-pipe-to-nvws --nvgpu-test-ws-lower-token %s | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#shared3 = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @pipe_to_mbarrier
  // CHECK-NOT: nvws.
  // CHECK: ttg.local_alloc
  // CHECK-SAME: !ttg.memdesc<2x1xi32
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<2x1xi64
  // CHECK-COUNT-4: ttng.init_barrier {{.*}}, 128
  // CHECK: gpu.barrier
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.arrive_barrier {{.*}}, 128 {async_task_id = array<i32: 0>, release_fence = true}
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 1>}
  // CHECK: ttg.local_load {{.*}} {async_task_id = array<i32: 1>}
  // CHECK: arith.cmpi ne
  // CHECK: ttng.arrive_barrier {{.*}}, 128 released[{{.*}}] ({{.*}}) {async_task_id = array<i32: 1>}
  tt.func @pipe_to_mbarrier(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%c1, %false] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tle.pipe.reader_release %a[%c1] {capacity = 2 : i32, pipe_name = "a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @pipe_same_task_to_mbarrier
  // CHECK-NOT: nvws.
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.arrive_barrier {{.*}}, 128 {async_task_id = array<i32: 0>, release_fence = true}
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.arrive_barrier {{.*}}, 128 released[{{.*}}] ({{.*}}) {async_task_id = array<i32: 0>}
  tt.func @pipe_same_task_to_mbarrier(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "same_task", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "same_task", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "same_task", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "same_task", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tle.pipe.reader_release %a[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "same_task", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @pipe_cpasync_to_mbarrier
  // CHECK-NOT: nvws.
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.async_copy_mbarrier_arrive {{.*}} {async_task_id = array<i32: 0>, noIncrement}
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 1>}
  tt.func @pipe_cpasync_to_mbarrier(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a_async", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a_async", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "a_async", field_names = ["a"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a_async", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tt.return
  }

  // CHECK-LABEL: @pipe_cpasync_same_task_to_mbarrier
  // CHECK-NOT: nvws.
  // CHECK: ttng.async_copy_mbarrier_arrive {{.*}} {async_task_id = array<i32: 0>, noIncrement}
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  tt.func @pipe_cpasync_same_task_to_mbarrier(%a: !ttg.memdesc<2x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a_async_same_task", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "a_async_same_task", field_names = ["a"], scope = "cta", tle.pipe_commit_cp_async} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "a_async_same_task", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x16xf16, #shared, #smem, mutable>
    scf.if %closed {
    }
    tt.return
  }

  // CHECK-LABEL: @pipe_tma_to_mbarrier
  // CHECK-NOT: nvws.
  // CHECK: ttng.init_barrier {{.*}}, 1
  // CHECK: ttng.init_barrier {{.*}}, 128
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.barrier_expect {{.*}}, 8192
  // CHECK: ttng.async_tma_copy_global_to_local
  // CHECK-NOT: ttng.arrive_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 1>}
  tt.func @pipe_tma_to_mbarrier(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 2 : i32, pipe_name = "a_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%c0, %false] {capacity = 2 : i32, pipe_name = "a_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    scf.if %closed {
    }
    tt.return
  }

  // CHECK-LABEL: @pipe_tma_same_task_to_mbarrier
  // CHECK-NOT: nvws.
  // CHECK: ttng.init_barrier {{.*}}, 1
  // CHECK: ttng.init_barrier {{.*}}, 128
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.barrier_expect {{.*}}, 8192
  // CHECK: ttng.async_tma_copy_global_to_local
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  tt.func @pipe_tma_same_task_to_mbarrier(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 2 : i32, pipe_name = "a_tma_same_task", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "a_tma_same_task", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "a_tma_same_task", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 2 : i32, pipe_name = "a_tma_same_task", field_names = ["a"], scope = "cta"} : !ttg.memdesc<2x32x64xf32, #nvmma, #smem, mutable>
    scf.if %closed {
    }
    tt.return
  }

  // CHECK-LABEL: @one_shot_tma_pipe_to_mbarrier
  // CHECK-NOT: nvws.
  // CHECK: ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
  // CHECK: ttng.init_barrier {{.*}}, 1
  // CHECK-NOT: ttng.init_barrier {{.*}}, 128
  // CHECK: ttng.barrier_expect {{.*}}, 8192
  // CHECK: ttng.async_tma_copy_global_to_local
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 1>}
  // CHECK: arith.constant {{.*}}false
  // CHECK-NOT: ttng.arrive_barrier
  tt.func @one_shot_tma_pipe_to_mbarrier(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 1 : i32, pipe_name = "one_shot_tma", field_names = ["a"], scope = "cta", one_shot = true} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {capacity = 1 : i32, pipe_name = "one_shot_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 1 : i32, pipe_name = "one_shot_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    %closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 1>, capacity = 1 : i32, pipe_name = "one_shot_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    scf.if %closed {
    }
    tle.pipe.reader_release %a[%c0] {async_task_id = array<i32: 1>, capacity = 1 : i32, pipe_name = "one_shot_tma", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @one_shot_tma_pipe_same_partition_reader
  // CHECK-NOT: nvws.
  // CHECK: ttng.init_barrier {{.*}}, 1
  // CHECK-NOT: ttng.init_barrier {{.*}}, 128
  // CHECK: ttng.barrier_expect {{.*}}, 8192
  // CHECK: ttng.async_tma_copy_global_to_local
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 0>}
  // CHECK: ttng.wait_barrier {{.*}} {async_task_id = array<i32: 1>}
  // CHECK: arith.constant {{.*}}false
  // CHECK-NOT: ttng.arrive_barrier
  tt.func @one_shot_tma_pipe_same_partition_reader(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %a: !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 1 : i32, pipe_name = "one_shot_tma_spmc", field_names = ["a"], readers = ["owner", "peer"], scope = "cta", one_shot = true} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_acquire %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 1 : i32, pipe_name = "one_shot_tma_spmc", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    %slot = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable> -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    ttg.tma_copy %desc, %slot, [%c0, %c0] : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {async_task_id = array<i32: 0>, capacity = 1 : i32, pipe_name = "one_shot_tma_spmc", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    %owner_closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 0>, capacity = 1 : i32, pipe_name = "one_shot_tma_spmc", field_names = ["a"], reader_name = "owner", scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    %peer_closed = tle.pipe.reader_wait %a[%c0, %false] {async_task_id = array<i32: 1>, capacity = 1 : i32, pipe_name = "one_shot_tma_spmc", field_names = ["a"], reader_name = "peer", scope = "cta"} : !ttg.memdesc<1x32x64xf32, #nvmma, #smem, mutable>
    scf.if %owner_closed {
    }
    scf.if %peer_closed {
    }
    tt.return
  }

  // CHECK-LABEL: @coalesce_adjacent_pipe_init_barriers
  // CHECK: ttng.init_barrier
  // CHECK: ttng.init_barrier
  // CHECK-COUNT-1: gpu.barrier
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.wait_barrier
  // CHECK: ttng.wait_barrier
  tt.func @coalesce_adjacent_pipe_init_barriers(%a: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 1 : i32, pipe_name = "coalesce_a", field_names = ["a"], scope = "cta", one_shot = true} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %payload = ttg.local_alloc : () -> !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %tag_zero = arith.constant 0 : i32
    %tag_init = tt.splat %tag_zero : i32 -> tensor<2x1xi32, #blocked2>
    %tag_alloc = ttg.local_alloc %tag_init : (tensor<2x1xi32, #blocked2>) -> !ttg.memdesc<2x1xi32, #shared, #smem, mutable>
    tle.pipe.create %b {capacity = 1 : i32, pipe_name = "coalesce_b", field_names = ["b"], scope = "cta", one_shot = true} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 1 : i32, pipe_name = "coalesce_a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %b[%c0] {capacity = 1 : i32, pipe_name = "coalesce_b", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %closed_a = tle.pipe.reader_wait %a[%c0, %false] {capacity = 1 : i32, pipe_name = "coalesce_a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %closed_b = tle.pipe.reader_wait %b[%c0, %false] {capacity = 1 : i32, pipe_name = "coalesce_b", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }

  // CHECK-LABEL: @keep_pipe_init_barrier_before_first_use
  // CHECK: ttng.init_barrier
  // CHECK: gpu.barrier
  // CHECK: ttng.arrive_barrier
  // CHECK: ttng.init_barrier
  // CHECK: gpu.barrier
  // CHECK: ttng.arrive_barrier
  tt.func @keep_pipe_init_barrier_before_first_use(%a: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>, %b: !ttg.memdesc<1x16xf16, #shared, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    %false = arith.constant false
    tle.pipe.create %a {capacity = 1 : i32, pipe_name = "ordered_a", field_names = ["a"], scope = "cta", one_shot = true} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %a[%c0] {capacity = 1 : i32, pipe_name = "ordered_a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.create %b {capacity = 1 : i32, pipe_name = "ordered_b", field_names = ["b"], scope = "cta", one_shot = true} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tle.pipe.writer_commit %b[%c0] {capacity = 1 : i32, pipe_name = "ordered_b", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %closed_a = tle.pipe.reader_wait %a[%c0, %false] {capacity = 1 : i32, pipe_name = "ordered_a", field_names = ["a"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    %closed_b = tle.pipe.reader_wait %b[%c0, %false] {capacity = 1 : i32, pipe_name = "ordered_b", field_names = ["b"], scope = "cta"} : !ttg.memdesc<1x16xf16, #shared, #smem, mutable>
    tt.return
  }
}
