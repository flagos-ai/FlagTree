// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(cuda_tile.module(cuda_tile.entry(auto-gen-memory-token{autogen-alias-memtoken=true})))" 2>/dev/null | FileCheck %s

// Test: basic same-root ordering. A load RAW-depends on the previous store,
// the following store WAR-depends on the load through the eager lastOp join,
// and the final store WAW-depends on the previous store.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_raw_war_waw(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %s0 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      %v, %l0 = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      %s1 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      %s2 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_raw_war_waw
// CHECK-SAME:  (%[[P:.+]]: tile<ptr<f32>>)
// CHECK:       %[[S0:.+]] = store_ptr_tko weak %[[P]],
// CHECK:       %{{.+}}, %[[L0:.+]] = load_ptr_tko weak %[[P]] token{{ ?}}={{ ?}}%[[S0]]
// CHECK:       %[[JOIN:.+]] = join_tokens %[[S0]], %[[L0]] : token
// CHECK:       %[[S1:.+]] = store_ptr_tko weak %[[P]], {{.*}} token{{ ?}}={{ ?}}%[[JOIN]]
// CHECK:       store_ptr_tko weak %[[P]], {{.*}} token{{ ?}}={{ ?}}%[[S1]]

// -----

// Test: a lone load with no fence/barrier has no ordering hazard, so the pass
// should leave it un-tokenized instead of materializing an unnecessary token
// graph.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_lone_load_noop(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %v0, %l0 = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_lone_load_noop
// CHECK-SAME:  (%[[P:.+]]: tile<ptr<f32>>)
// CHECK-NOT:   make_token
// CHECK-NOT:   join_tokens
// CHECK:       load_ptr_tko weak %[[P]] : tile<ptr<f32>> -> tile<f32>, token
// CHECK-NOT:   token{{ ?}}=
// CHECK:       return

// -----

// Test: a lone store with no fence/barrier also has no ordering hazard; there
// is no later operation that can observe a token edge.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_lone_store_noop(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %s0 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_lone_store_noop
// CHECK-SAME:  (%[[P:.+]]: tile<ptr<f32>>)
// CHECK-NOT:   make_token
// CHECK-NOT:   join_tokens
// CHECK:       store_ptr_tko weak %[[P]], {{.*}} : tile<ptr<f32>>, tile<f32> -> token
// CHECK-NOT:   token{{ ?}}=
// CHECK:       return

// -----

// Test: repeated reads of the same root do not form a read-after-read token
// chain. Each read has no input token when there is no prior store, while the
// eager lastOp joins accumulate those read outputs for a later WAR store.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_read_read_no_chain(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %v0, %l0 = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      %v1, %l1 = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      %v2, %l2 = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      %s0 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_read_read_no_chain
// CHECK-SAME:  (%[[P:.+]]: tile<ptr<f32>>)
// CHECK:       %{{.+}}, %[[L0:.+]] = load_ptr_tko weak %[[P]] : tile<ptr<f32>> -> tile<f32>, token
// CHECK:       %{{.+}}, %[[L1:.+]] = load_ptr_tko weak %[[P]] : tile<ptr<f32>> -> tile<f32>, token
// CHECK:       %[[JOIN01:.+]] = join_tokens %[[L0]], %[[L1]] : token
// CHECK:       %{{.+}}, %[[L2:.+]] = load_ptr_tko weak %[[P]] : tile<ptr<f32>> -> tile<f32>, token
// CHECK:       %[[JOIN012:.+]] = join_tokens %[[JOIN01]], %[[L2]] : token
// CHECK:       store_ptr_tko weak %[[P]], {{.*}} token{{ ?}}={{ ?}}%[[JOIN012]]

// -----

// Test: distinct roots do not accidentally chain through each other. A store
// on A does not become the input token for a load on B; only the later store
// on B consumes the B-load output token.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_independent_roots(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %s0 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      %v1, %l1 = cuda_tile.load_ptr_tko weak %arg1 : tile<ptr<f32>> -> tile<f32>, token
      %s1 = cuda_tile.store_ptr_tko weak %arg1, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_independent_roots
// CHECK-SAME:  (%[[A:.+]]: tile<ptr<f32>>, %[[B:.+]]: tile<ptr<f32>>)
// CHECK:       store_ptr_tko weak %[[A]],
// CHECK:       %{{.+}}, %[[LB:.+]] = load_ptr_tko weak %[[B]] : tile<ptr<f32>> -> tile<f32>, token
// CHECK:       store_ptr_tko weak %[[B]], {{.*}} token{{ ?}}={{ ?}}%[[LB]]


// -----

// Same backing pointer through a view: atomic reduction must wait for the prior
// store, and the following load must wait for the atomic reduction.
module {
  cuda_tile.module @m {
    entry @descriptor_atomic_token_order(%ptr: tile<ptr<i32>>) {
      %value = constant <i32: 1> : tile<i32>
      %s = store_ptr_tko weak %ptr, %value : tile<ptr<i32>>, tile<i32> -> token
      %tv = make_tensor_view %ptr, shape=[8], strides=[1] : tensor_view<8xi32, strides=[1]>
      %view = make_strided_view %tv : strided_view<tile=(8), traversal_strides=[1], tensor_view<8xi32, strides=[1]>>
      %zero = constant <i32: 0> : tile<i32>
      %tile = constant <i32: 1> : tile<8xi32>
      %a = atomic_red_view_tko relaxed device %view[%zero], add, %tile : tile<8xi32>, strided_view<tile=(8), traversal_strides=[1], tensor_view<8xi32, strides=[1]>>, tile<i32> -> token
      %v, %l = load_ptr_tko weak %ptr : tile<ptr<i32>> -> tile<i32>, token
      return
    }
  }
}
// CHECK-LABEL: @descriptor_atomic_token_order
// CHECK: %[[STORE:.*]] = store_ptr_tko weak
// CHECK: %[[ATOMIC:.*]] = atomic_red_view_tko relaxed device {{.*}} token{{ ?}}={{ ?}}%[[STORE]]
// CHECK: load_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[ATOMIC]]
