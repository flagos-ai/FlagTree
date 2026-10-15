// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(cuda_tile.module(cuda_tile.entry(auto-gen-memory-token{autogen-alias-memtoken=true})))" 2>/dev/null | FileCheck %s

// Test: gpu.barrier is a total fence. A store to root A before the barrier
// must order a later load from otherwise untouched root B; the pass erases the
// barrier and carries that ordering through tokens.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_barrier_fanout_to_untouched_root(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %s0 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      gpu.barrier
      %v1, %l1 = cuda_tile.load_ptr_tko weak %arg1 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_barrier_fanout_to_untouched_root
// CHECK-SAME:  (%[[A:.+]]: tile<ptr<f32>>, %[[B:.+]]: tile<ptr<f32>>)
// CHECK:       %[[S0:.+]] = store_ptr_tko weak %[[A]],
// CHECK-NOT:   gpu.barrier
// CHECK:       load_ptr_tko weak %[[B]] token{{ ?}}={{ ?}}%[[S0]]

// -----

// Test: gdc_wait_tko is modeled as an acquire fence. It does not consume prior
// local memory state, but its result orders later memory ops.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_gdc_wait_orders_later_load(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %v0, %l0 = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      %wait = cuda_tile.gdc_wait_tko -> token
      %v1, %l1 = cuda_tile.load_ptr_tko weak %arg1 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_gdc_wait_orders_later_load
// CHECK-SAME:  (%[[A:.+]]: tile<ptr<f32>>, %[[B:.+]]: tile<ptr<f32>>)
// CHECK:       load_ptr_tko weak %[[A]] : tile<ptr<f32>> -> tile<f32>, token
// CHECK:       %[[WAIT:.+]] = gdc_wait_tko -> token
// CHECK:       load_ptr_tko weak %[[B]] token{{ ?}}={{ ?}}%[[WAIT]]

// -----

// Test: gdc_launch_dependents_tko is a dependent-launch signal. It consumes
// prior producer writes, but not read-only tokens, and does not broadcast its
// output to later unrelated memory ops.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_gdc_launch_anchors_on_prior_memory(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>, %arg2: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %v0, %l0 = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      %s0 = cuda_tile.store_ptr_tko weak %arg1, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      %launch = cuda_tile.gdc_launch_dependents_tko -> token
      %v1, %l1 = cuda_tile.load_ptr_tko weak %arg2 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_gdc_launch_anchors_on_prior_memory
// CHECK-SAME:  (%[[A:.+]]: tile<ptr<f32>>, %[[B:.+]]: tile<ptr<f32>>, %[[C:.+]]: tile<ptr<f32>>)
// CHECK:       %[[CST:.+]] = constant <f32: 1.000000e+00> : tile<f32>
// CHECK:       load_ptr_tko weak %[[A]] : tile<ptr<f32>> -> tile<f32>, token
// CHECK:       %[[S0:.+]] = store_ptr_tko weak %[[B]], %[[CST]]
// CHECK-NEXT:  %[[LAUNCH:.+]] = gdc_launch_dependents_tko token{{ ?}}={{ ?}}%[[S0]] -> token
// CHECK-NEXT:  load_ptr_tko weak %[[C]] : tile<ptr<f32>> -> tile<f32>, token

// -----

// Test: a user-tokenized gdc_wait_tko makes the function manually tokenized.
// The pass must not append another token to the fence, and must not auto-tokenize
// otherwise untokenized memory ops in the same body.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_manual_gdc_wait_bails_out(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %manual = cuda_tile.make_token : token
      %s0 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      %wait = cuda_tile.gdc_wait_tko token = %manual -> token
      %v0, %l0 = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_manual_gdc_wait_bails_out
// CHECK-SAME:  (%[[P:.+]]: tile<ptr<f32>>)
// CHECK:       %[[CST:.+]] = constant <f32: 1.000000e+00> : tile<f32>
// CHECK:       %[[MANUAL:.+]] = make_token
// CHECK:       store_ptr_tko weak %[[P]], %[[CST]] : tile<ptr<f32>>, tile<f32> -> token
// CHECK:       %[[WAIT:.+]] = gdc_wait_tko token{{ ?}}={{ ?}}%[[MANUAL]] -> token
// CHECK:       load_ptr_tko weak %[[P]] : tile<ptr<f32>> -> tile<f32>, token

// -----

// Test: same manual-token bailout for dependent-launch signals.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_manual_gdc_launch_bails_out(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %manual = cuda_tile.make_token : token
      %v0, %l0 = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      %launch = cuda_tile.gdc_launch_dependents_tko token = %manual -> token
      %s0 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_manual_gdc_launch_bails_out
// CHECK-SAME:  (%[[P:.+]]: tile<ptr<f32>>)
// CHECK:       %[[CST:.+]] = constant <f32: 1.000000e+00> : tile<f32>
// CHECK:       %[[MANUAL:.+]] = make_token
// CHECK:       load_ptr_tko weak %[[P]] : tile<ptr<f32>> -> tile<f32>, token
// CHECK:       %[[LAUNCH:.+]] = gdc_launch_dependents_tko token{{ ?}}={{ ?}}%[[MANUAL]] -> token
// CHECK:       store_ptr_tko weak %[[P]], %[[CST]] : tile<ptr<f32>>, tile<f32> -> token

// -----

// Test: atomics are memory effects. Repeated atomics to the same root must be
// ordered by token dependencies, even when there are no normal loads/stores
// between them.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_atomic_rmw_same_root(%arg0: tile<ptr<i32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %v0, %a0 = cuda_tile.atomic_rmw_tko relaxed device %arg0, add, %cst_1_i32 : tile<ptr<i32>>, tile<i32> -> tile<i32>, token
      %v1, %a1 = cuda_tile.atomic_rmw_tko relaxed device %arg0, add, %cst_1_i32 : tile<ptr<i32>>, tile<i32> -> tile<i32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_atomic_rmw_same_root
// CHECK-SAME:  (%[[P:.+]]: tile<ptr<i32>>)
// CHECK:       %{{.+}}, %[[A0:.+]] = atomic_rmw_tko relaxed device %[[P]], add,
// CHECK:       atomic_rmw_tko relaxed device %[[P]], add, {{.*}} token{{ ?}}={{ ?}}%[[A0]]
