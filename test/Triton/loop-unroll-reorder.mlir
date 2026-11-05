// RUN: triton-opt --split-input-file %s -triton-loop-unroll | FileCheck %s

// reorder-loop-loads (tle.range(..., reorder=...)), applied by the LoopUnroll pass.

// reorder = 1: the loads of the next iteration are issued before the compute of
// the current one, the loads of later iterations are not.
// CHECK-LABEL: @reorder_distance_one
// CHECK-NOT: scf.for
// CHECK: tt.load
// CHECK-NOT: tt.store
// CHECK: tt.load
// CHECK-NOT: tt.load
// CHECK: tt.store
// CHECK-NOT: tt.store
// CHECK: tt.load
// CHECK-NOT: tt.load
// CHECK: tt.store
// CHECK-NOT: tt.store
// CHECK: tt.load
// CHECK-NOT: tt.load
// CHECK: tt.store
// CHECK-NOT: tt.load
// CHECK: tt.store
// CHECK-NOT: tle.reorder_iteration
// CHECK-NOT: tt.reorder
tt.func @reorder_distance_one(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c4 step %c1 : i32 {
    %p = tt.addptr %src, %i : !tt.ptr<f32>, i32
    %v = tt.load %p : !tt.ptr<f32>
    %w = arith.addf %v, %cst : f32
    %q = tt.addptr %dst, %i : !tt.ptr<f32>, i32
    tt.store %q, %w : !tt.ptr<f32>
  } {tt.loop_unroll_factor = 4 : i32, tt.reorder = 1 : i32}
  tt.return
}

// -----

// reorder = true: every load goes first, but ops after the loop stay after it:
// the load of %dst that follows the loop must still observe the loop's stores.
// CHECK-LABEL: @reorder_all_keeps_ops_after_loop
// CHECK-COUNT-4: tt.load %{{.*}} : !tt.ptr<f32>
// CHECK-NOT: tt.load
// CHECK-COUNT-4: tt.store
// CHECK: %[[AFTER:.*]] = tt.load %arg1 : !tt.ptr<f32>
// CHECK: tt.store %arg2, %[[AFTER]]
tt.func @reorder_all_keeps_ops_after_loop(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c4 step %c1 : i32 {
    %p = tt.addptr %src, %i : !tt.ptr<f32>, i32
    %v = tt.load %p : !tt.ptr<f32>
    %w = arith.addf %v, %cst : f32
    %q = tt.addptr %dst, %i : !tt.ptr<f32>, i32
    tt.store %q, %w : !tt.ptr<f32>
  } {tt.loop_unroll_factor = 4 : i32, tt.reorder = true}
  %after = tt.load %dst : !tt.ptr<f32>
  tt.store %out, %after : !tt.ptr<f32>
  tt.return
}

// -----

// A load that reads what its own iteration stored is not moved.
// CHECK-LABEL: @reorder_keeps_load_after_own_store
// CHECK: tt.store %{{.*}}, %arg2
// CHECK: tt.store %{{.*}}, %arg2
// CHECK-NOT: tt.store
// CHECK: tt.load
// CHECK-NOT: tt.load
// CHECK: tt.store
// CHECK-NOT: tt.load
// CHECK: tt.store %{{.*}}, %arg2
// CHECK-NOT: tt.store
// CHECK: tt.load
// CHECK: tt.store
tt.func @reorder_keeps_load_after_own_store(%buf: !tt.ptr<f32>, %out: !tt.ptr<f32>, %x: f32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %p0 = tt.addptr %buf, %c0 : !tt.ptr<f32>, i32
  %o0 = tt.addptr %out, %c0 : !tt.ptr<f32>, i32
  tt.store %p0, %x : !tt.ptr<f32>
  scf.for %i = %c0 to %c2 step %c1 : i32 {
    %p = tt.addptr %buf, %i : !tt.ptr<f32>, i32
    %o = tt.addptr %out, %i : !tt.ptr<f32>, i32
    tt.store %p, %x : !tt.ptr<f32>
    %v = tt.load %p : !tt.ptr<f32>
    tt.store %o, %v : !tt.ptr<f32>
  } {tt.loop_unroll_factor = 2 : i32, tt.reorder = true}
  tt.return
}

// -----

// No load is moved across an atomic of an earlier iteration.
// CHECK-LABEL: @reorder_stops_at_atomic
// CHECK: tt.load
// CHECK-NOT: tt.load
// CHECK: tt.atomic_rmw
// CHECK-NOT: tt.load
// CHECK: tt.store
// CHECK: tt.load
// CHECK-NOT: tt.load
// CHECK: tt.atomic_rmw
// CHECK: tt.store
tt.func @reorder_stops_at_atomic(%src: !tt.ptr<f32>, %cnt: !tt.ptr<f32>, %dst: !tt.ptr<f32>) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %c2 step %c1 : i32 {
    %p = tt.addptr %src, %i : !tt.ptr<f32>, i32
    %q = tt.addptr %dst, %i : !tt.ptr<f32>, i32
    %v = tt.load %p : !tt.ptr<f32>
    %old = tt.atomic_rmw fadd, acq_rel, gpu, %cnt, %cst : (!tt.ptr<f32>, f32) -> f32
    %w = arith.addf %v, %old : f32
    tt.store %q, %w : !tt.ptr<f32>
  } {tt.loop_unroll_factor = 2 : i32, tt.reorder = true}
  tt.return
}

// -----

// Partial unroll: the loads are reordered inside the unrolled main loop, the
// epilogue loop is left alone.
// CHECK-LABEL: @reorder_partial_unroll
// CHECK: scf.for
// CHECK-COUNT-2: tt.load
// CHECK-NOT: tt.load
// CHECK-COUNT-2: tt.store
// CHECK: scf.for
// CHECK: tt.load
// CHECK-NOT: tt.load
// CHECK: tt.store
// CHECK: tt.num_stages = 1 : i32
tt.func @reorder_partial_unroll(%src: !tt.ptr<f32>, %dst: !tt.ptr<f32>, %n: i32) {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %cst = arith.constant 1.000000e+00 : f32
  scf.for %i = %c0 to %n step %c1 : i32 {
    %p = tt.addptr %src, %i : !tt.ptr<f32>, i32
    %v = tt.load %p : !tt.ptr<f32>
    %w = arith.addf %v, %cst : f32
    %q = tt.addptr %dst, %i : !tt.ptr<f32>, i32
    tt.store %q, %w : !tt.ptr<f32>
  } {tt.loop_unroll_factor = 2 : i32, tt.reorder = true}
  tt.return
}
