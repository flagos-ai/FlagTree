// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(cuda_tile.module(cuda_tile.entry(auto-gen-memory-token{autogen-alias-memtoken=true})))" 2>/dev/null | FileCheck %s

// Tests for the per-iteration disjoint-store proof from
// TkoDependenceAnalysis::isLoopCarriedStoreDisjoint. AutoGenMemoryToken
// consumes the proof to elide the per-iter WAW edge on lastStore[c] and use
// a loop-invariant root token instead.
//
// Patterns covered:
//   1. partition_view + IV index   → proof (existing behavior, unchanged)
//   2. strided_view + IV index, traversal_strides[d] >= tile_shape[d]
//                                  → proof
//   3. strided_view + IV index, traversal_strides[d] <  tile_shape[d]
//                                  → no proof (footprints overlap; this
//                                    closes a latent unsoundness in the
//                                    pre-Stage-0 matcher)
//      These view proofs are rank-generic; the 3-D cases below anchor that the
//      implementation is not accidentally 2-D-only.
//   4. store_ptr_tko + affine offset where IV stride dominates lane interval
//                                  → proof
//   5. store_ptr_tko + 2-D affine offset where a finite per-lane offset set
//      is disjoint from all loop-shifted copies reachable by this loop
//                                  → proof
//   6. store_ptr_tko + affine offset where IV stride is smaller than lane
//      interval, or non-affine dynamic stride → no proof
//   7. same-root load + disjoint store in one loop body → no proof consumer
//
// Test: strided_view with traversal_strides == tile_shape (disjoint per-iter
// footprint) should fold to the root token, mirroring the partition_view
// happy path.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_strided_view_disjoint_iv_store(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64x64xf32>
      %tview = cuda_tile.make_tensor_view %arg0, shape=[256, 256], strides=[256, 1] : tensor_view<256x256xf32, strides=[256,1]>
      %sview = cuda_tile.make_strided_view %tview : strided_view<tile=(64x64), traversal_strides=[64, 64], tensor_view<256x256xf32, strides=[256,1]>>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %store = cuda_tile.store_view_tko weak %tile, %sview[%i, %cst_0_i32] : tile<64x64xf32>, strided_view<tile=(64x64), traversal_strides=[64, 64], tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_view_tko weak %sview[%cst_0_i32, %cst_0_i32] : strided_view<tile=(64x64), traversal_strides=[64, 64], tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> tile<64x64xf32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_strided_view_disjoint_iv_store
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values({{.*}}) -> (token, token) {
// CHECK:   %[[STORE:.*]] = store_view_tko weak {{.*}}[%[[IV]], {{.*}}] token{{ ?}}={{ ?}}%[[ROOT]]
// CHECK:   continue {{.*}} : token, token
// CHECK: }
// CHECK: load_view_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: the same strided_view proof is rank-generic. In this 3-D case, the
// loop IV indexes dim 0 and traversal_strides[0] == tile_shape[0], so each
// iteration stores a disjoint 8x8x8 tile.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_strided_view_3d_disjoint_iv_store(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<8x8x8xf32>
      %tview = cuda_tile.make_tensor_view %arg0, shape=[128, 128, 128], strides=[16384, 128, 1] : tensor_view<128x128x128xf32, strides=[16384,128,1]>
      %sview = cuda_tile.make_strided_view %tview : strided_view<tile=(8x8x8), traversal_strides=[8, 8, 8], tensor_view<128x128x128xf32, strides=[16384,128,1]>>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %store = cuda_tile.store_view_tko weak %tile, %sview[%i, %cst_0_i32, %cst_0_i32] : tile<8x8x8xf32>, strided_view<tile=(8x8x8), traversal_strides=[8, 8, 8], tensor_view<128x128x128xf32, strides=[16384,128,1]>>, tile<i32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_view_tko weak %sview[%cst_0_i32, %cst_0_i32, %cst_0_i32] : strided_view<tile=(8x8x8), traversal_strides=[8, 8, 8], tensor_view<128x128x128xf32, strides=[16384,128,1]>>, tile<i32> -> tile<8x8x8xf32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_strided_view_3d_disjoint_iv_store
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values({{.*}}) -> (token, token) {
// CHECK:   %[[STORE:.*]] = store_view_tko weak {{.*}}[%[[IV]], {{.*}}, {{.*}}] token{{ ?}}={{ ?}}%[[ROOT]]
// CHECK:   continue {{.*}} : token, token
// CHECK: }
// CHECK: load_view_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: 3-D strided_view overlap on the IV-indexed dim. The loop advances
// by 4 elements on dim 0, but each iteration stores an 8-wide tile there, so
// adjacent iterations overlap and the WAW edge must stay loop-carried.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_strided_view_3d_overlapping_iv_store(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<8x8x8xf32>
      %tview = cuda_tile.make_tensor_view %arg0, shape=[128, 128, 128], strides=[16384, 128, 1] : tensor_view<128x128x128xf32, strides=[16384,128,1]>
      %sview = cuda_tile.make_strided_view %tview : strided_view<tile=(8x8x8), traversal_strides=[4, 8, 8], tensor_view<128x128x128xf32, strides=[16384,128,1]>>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %store = cuda_tile.store_view_tko weak %tile, %sview[%i, %cst_0_i32, %cst_0_i32] : tile<8x8x8xf32>, strided_view<tile=(8x8x8), traversal_strides=[4, 8, 8], tensor_view<128x128x128xf32, strides=[16384,128,1]>>, tile<i32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_view_tko weak %sview[%cst_0_i32, %cst_0_i32, %cst_0_i32] : strided_view<tile=(8x8x8), traversal_strides=[4, 8, 8], tensor_view<128x128x128xf32, strides=[16384,128,1]>>, tile<i32> -> tile<8x8x8xf32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_strided_view_3d_overlapping_iv_store
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT]], {{.*}}) -> (token, token) {
// CHECK:   %[[STORE:.*]] = store_view_tko weak {{.*}} token{{ ?}}={{ ?}}%[[LASTOP]]
// CHECK:   continue %[[STORE]], %[[STORE]] : token, token
// CHECK: }
// CHECK: load_view_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: a same-root load blocks the parallel-store token optimization. The
// partition_view store is independently loop-carried disjoint, but
// AutoGenMemoryToken must keep the normal lastOp/lastStore chain because the
// root has an in-loop read.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_same_root_load_blocks_parallel_store(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64x64xf32>
      %tview = cuda_tile.make_tensor_view %arg0, shape=[256, 256], strides=[256, 1] : tensor_view<256x256xf32, strides=[256,1]>
      %pview = cuda_tile.make_partition_view %tview : partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %loaded, %load_token = cuda_tile.load_view_tko weak %pview[%i, %cst_0_i32] : partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> tile<64x64xf32>, token
        %store = cuda_tile.store_view_tko weak %tile, %pview[%i, %cst_0_i32] : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> token
        cuda_tile.continue
      }
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_same_root_load_blocks_parallel_store
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[ACC_INIT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT]], %[[LASTSTORE:.*]] = %[[ACC_INIT]]) -> (token, token) {
// CHECK:   %{{.*}}, %[[LOAD:.*]] = load_view_tko weak {{.*}}[%[[IV]], {{.*}}] token{{ ?}}={{ ?}}%[[LASTSTORE]]
// CHECK:   %[[READ_JOIN:.*]] = join_tokens %[[LASTOP]], %[[LOAD]] : token
// CHECK:   %[[STORE:.*]] = store_view_tko weak {{.*}}[%[[IV]], {{.*}}] token{{ ?}}={{ ?}}%[[READ_JOIN]]
// CHECK:   continue %[[STORE]], %[[STORE]] : token, token
// CHECK: }

// -----

// Test: store_ptr_tko with pointer offsets i * 64 + iota(64). The lane
// interval is [0,63] and adjacent loop iterations are separated by 64
// elements, so the stores are disjoint across iterations and may use the
// loop-invariant root token.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_store_ptr_affine_disjoint_iv_store(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %cst_64_i32 = cuda_tile.constant <i32: 64> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64xf32>
      %base1 = cuda_tile.reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base = cuda_tile.broadcast %base1 : tile<1xptr<f32>> -> tile<64xptr<f32>>
      %iota = cuda_tile.iota : tile<64xi32>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %iv_scaled = cuda_tile.muli %i, %cst_64_i32 : tile<i32>
        %iv_scaled1 = cuda_tile.reshape %iv_scaled : tile<i32> -> tile<1xi32>
        %iv_vec = cuda_tile.broadcast %iv_scaled1 : tile<1xi32> -> tile<64xi32>
        %offset = cuda_tile.addi %iota, %iv_vec : tile<64xi32>
        %ptrs = cuda_tile.offset %base, %offset : tile<64xptr<f32>>, tile<64xi32> -> tile<64xptr<f32>>
        %store = cuda_tile.store_ptr_tko weak %ptrs, %tile : tile<64xptr<f32>>, tile<64xf32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_store_ptr_affine_disjoint_iv_store
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[ACC_INIT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT]], %[[LASTSTORE:.*]] = %[[ACC_INIT]]) -> (token, token) {
// CHECK:   %[[STORE:.*]] = store_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[ROOT]]
// CHECK:   %[[JOIN:.*]] = join_tokens %[[LASTSTORE]], %[[STORE]] : token
// CHECK:   continue %[[JOIN]], %[[JOIN]] : token, token
// CHECK: }
// CHECK: load_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: store_ptr_tko with a 2-D pointer tile matching the Triton
// out[row, n_idx * 64 + col] lowering shape. The row stride is 256 and the
// loop has four 64-wide column blocks, so shifting the 64x64 lane set by any
// reachable iteration delta never reaches the next row.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_store_ptr_2d_affine_disjoint_iv_store(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %cst_64_i32 = cuda_tile.constant <i32: 64> : tile<i32>
      %cst_256_i32 = cuda_tile.constant <i32: 256> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64x64xf32>
      %base1 = cuda_tile.reshape %arg0 : tile<ptr<f32>> -> tile<1x1xptr<f32>>
      %base = cuda_tile.broadcast %base1 : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
      %row_iota = cuda_tile.iota : tile<64xi32>
      %row_iota1 = cuda_tile.reshape %row_iota : tile<64xi32> -> tile<64x1xi32>
      %row = cuda_tile.broadcast %row_iota1 : tile<64x1xi32> -> tile<64x64xi32>
      %col_iota = cuda_tile.iota : tile<64xi32>
      %col_iota1 = cuda_tile.reshape %col_iota : tile<64xi32> -> tile<1x64xi32>
      %col = cuda_tile.broadcast %col_iota1 : tile<1x64xi32> -> tile<64x64xi32>
      %row_stride1 = cuda_tile.reshape %cst_256_i32 : tile<i32> -> tile<1x1xi32>
      %row_stride = cuda_tile.broadcast %row_stride1 : tile<1x1xi32> -> tile<64x64xi32>
      %row_offset = cuda_tile.muli %row, %row_stride : tile<64x64xi32>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %iv_scaled = cuda_tile.muli %i, %cst_64_i32 : tile<i32>
        %iv_scaled1 = cuda_tile.reshape %iv_scaled : tile<i32> -> tile<1x1xi32>
        %iv_scaled2 = cuda_tile.broadcast %iv_scaled1 : tile<1x1xi32> -> tile<64x64xi32>
        %col_offset = cuda_tile.addi %col, %iv_scaled2 : tile<64x64xi32>
        %offset = cuda_tile.addi %row_offset, %col_offset : tile<64x64xi32>
        %ptrs = cuda_tile.offset %base, %offset : tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>
        %store = cuda_tile.store_ptr_tko weak %ptrs, %tile : tile<64x64xptr<f32>>, tile<64x64xf32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_store_ptr_2d_affine_disjoint_iv_store
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[ACC_INIT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT]], %[[LASTSTORE:.*]] = %[[ACC_INIT]]) -> (token, token) {
// CHECK:   %[[STORE:.*]] = store_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[ROOT]]
// CHECK:   %[[JOIN:.*]] = join_tokens %[[LASTSTORE]], %[[STORE]] : token
// CHECK:   continue %[[JOIN]], %[[JOIN]] : token, token
// CHECK: }
// CHECK: load_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: the same 2-D pointer pattern with row_stride=128 and four 64-wide
// column blocks can collide across iterations: row 1 at iter 0 aliases row 0
// at iter 2. The WAW chain must be preserved.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_store_ptr_2d_affine_row_collision(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %cst_64_i32 = cuda_tile.constant <i32: 64> : tile<i32>
      %cst_128_i32 = cuda_tile.constant <i32: 128> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64x64xf32>
      %base1 = cuda_tile.reshape %arg0 : tile<ptr<f32>> -> tile<1x1xptr<f32>>
      %base = cuda_tile.broadcast %base1 : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
      %row_iota = cuda_tile.iota : tile<64xi32>
      %row_iota1 = cuda_tile.reshape %row_iota : tile<64xi32> -> tile<64x1xi32>
      %row = cuda_tile.broadcast %row_iota1 : tile<64x1xi32> -> tile<64x64xi32>
      %col_iota = cuda_tile.iota : tile<64xi32>
      %col_iota1 = cuda_tile.reshape %col_iota : tile<64xi32> -> tile<1x64xi32>
      %col = cuda_tile.broadcast %col_iota1 : tile<1x64xi32> -> tile<64x64xi32>
      %row_stride1 = cuda_tile.reshape %cst_128_i32 : tile<i32> -> tile<1x1xi32>
      %row_stride = cuda_tile.broadcast %row_stride1 : tile<1x1xi32> -> tile<64x64xi32>
      %row_offset = cuda_tile.muli %row, %row_stride : tile<64x64xi32>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %iv_scaled = cuda_tile.muli %i, %cst_64_i32 : tile<i32>
        %iv_scaled1 = cuda_tile.reshape %iv_scaled : tile<i32> -> tile<1x1xi32>
        %iv_scaled2 = cuda_tile.broadcast %iv_scaled1 : tile<1x1xi32> -> tile<64x64xi32>
        %col_offset = cuda_tile.addi %col, %iv_scaled2 : tile<64x64xi32>
        %offset = cuda_tile.addi %row_offset, %col_offset : tile<64x64xi32>
        %ptrs = cuda_tile.offset %base, %offset : tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>
        %store = cuda_tile.store_ptr_tko weak %ptrs, %tile : tile<64x64xptr<f32>>, tile<64x64xf32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_store_ptr_2d_affine_row_collision
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT]], {{.*}}) -> (token, token) {
// CHECK:   %[[STORE:.*]] = store_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[LASTOP]]
// CHECK:   continue %[[STORE]], %[[STORE]] : token, token
// CHECK: }
// CHECK: load_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: store_ptr_tko with pointer offsets i + iota(64). Adjacent iterations
// overlap on 63 of 64 lanes, so the WAW chain through loop-carried lastOp must
// be preserved.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_store_ptr_affine_overlapping_iv_store(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64xf32>
      %base1 = cuda_tile.reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base = cuda_tile.broadcast %base1 : tile<1xptr<f32>> -> tile<64xptr<f32>>
      %iota = cuda_tile.iota : tile<64xi32>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %iv1 = cuda_tile.reshape %i : tile<i32> -> tile<1xi32>
        %iv_vec = cuda_tile.broadcast %iv1 : tile<1xi32> -> tile<64xi32>
        %offset = cuda_tile.addi %iota, %iv_vec : tile<64xi32>
        %ptrs = cuda_tile.offset %base, %offset : tile<64xptr<f32>>, tile<64xi32> -> tile<64xptr<f32>>
        %store = cuda_tile.store_ptr_tko weak %ptrs, %tile : tile<64xptr<f32>>, tile<64xf32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_store_ptr_affine_overlapping_iv_store
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT]], {{.*}}) -> (token, token) {
// CHECK:   %[[STORE:.*]] = store_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[LASTOP]]
// CHECK:   continue %[[STORE]], %[[STORE]] : token, token
// CHECK: }
// CHECK: load_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: store_ptr_tko with dynamic IV stride. The expression is affine only
// if the multiplier is a compile-time constant; dynamic x IV is unsupported
// and must fall back to the conservative WAW chain.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_store_ptr_dynamic_stride_unsupported(%arg0: tile<ptr<f32>>, %scale: tile<i32>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64xf32>
      %base1 = cuda_tile.reshape %arg0 : tile<ptr<f32>> -> tile<1xptr<f32>>
      %base = cuda_tile.broadcast %base1 : tile<1xptr<f32>> -> tile<64xptr<f32>>
      %iota = cuda_tile.iota : tile<64xi32>
      %scale1 = cuda_tile.reshape %scale : tile<i32> -> tile<1xi32>
      %scale_vec = cuda_tile.broadcast %scale1 : tile<1xi32> -> tile<64xi32>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %iv1 = cuda_tile.reshape %i : tile<i32> -> tile<1xi32>
        %iv_vec = cuda_tile.broadcast %iv1 : tile<1xi32> -> tile<64xi32>
        %dynamic_stride = cuda_tile.muli %iv_vec, %scale_vec : tile<64xi32>
        %offset = cuda_tile.addi %iota, %dynamic_stride : tile<64xi32>
        %ptrs = cuda_tile.offset %base, %offset : tile<64xptr<f32>>, tile<64xi32> -> tile<64xptr<f32>>
        %store = cuda_tile.store_ptr_tko weak %ptrs, %tile : tile<64xptr<f32>>, tile<64xf32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_store_ptr_dynamic_stride_unsupported
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT]], {{.*}}) -> (token, token) {
// CHECK:   %[[STORE:.*]] = store_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[LASTOP]]
// CHECK:   continue %[[STORE]], %[[STORE]] : token, token
// CHECK: }
// CHECK: load_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: strided_view with traversal_strides < tile_shape on the IV-indexed
// dim. Adjacent iterations overlap on that dim, so the WAW chain through
// loop-carried lastOp must be preserved. (Soundness fix — the pre-Stage-0
// matcher accepted this case based on "any IV-derived index" and would have
// granted a proof here, dropping the WAW edge unsoundly.)

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_strided_view_overlapping_iv_store(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64x64xf32>
      %tview = cuda_tile.make_tensor_view %arg0, shape=[256, 256], strides=[256, 1] : tensor_view<256x256xf32, strides=[256,1]>
      // tile_shape=(64,64), traversal_strides=[32,64] — IV at dim 0 advances
      // by 32 each iter but each iter covers 64 elems on dim 0 → 32 elems
      // of overlap between adjacent iterations.
      %sview = cuda_tile.make_strided_view %tview : strided_view<tile=(64x64), traversal_strides=[32, 64], tensor_view<256x256xf32, strides=[256,1]>>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %store = cuda_tile.store_view_tko weak %tile, %sview[%i, %cst_0_i32] : tile<64x64xf32>, strided_view<tile=(64x64), traversal_strides=[32, 64], tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_view_tko weak %sview[%cst_0_i32, %cst_0_i32] : strided_view<tile=(64x64), traversal_strides=[32, 64], tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> tile<64x64xf32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_strided_view_overlapping_iv_store
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT]], {{.*}}) -> (token, token) {
// CHECK:   %[[STORE:.*]] = store_view_tko weak {{.*}} token{{ ?}}={{ ?}}%[[LASTOP]]
// CHECK:   continue %[[STORE]], %[[STORE]] : token, token
// CHECK: }
// CHECK: load_view_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: GDC wait/launch around a disjoint store loop. The acquire wait must
// still order the loop-body stores, but the disjoint proof must avoid the
// per-iteration WAW edge through the loop-carried lastOp token. This anchors
// the combined memory ordering objective: keep fence ordering while avoiding the
// over-serialized store chain.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_gdc_wait_disjoint_store_loop_before_launch(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64x64xf32>
      %tview = cuda_tile.make_tensor_view %arg0, shape=[256, 256], strides=[256, 1] : tensor_view<256x256xf32, strides=[256,1]>
      %pview = cuda_tile.make_partition_view %tview : partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>
      %wait = cuda_tile.gdc_wait_tko -> token
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %store = cuda_tile.store_view_tko weak %tile, %pview[%i, %cst_0_i32] : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> token
        cuda_tile.continue
      }
      %launch = cuda_tile.gdc_launch_dependents_tko -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_gdc_wait_disjoint_store_loop_before_launch
// CHECK: %[[WAIT:.*]] = gdc_wait_tko{{.*}}-> token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT:.*]], %[[LASTSTORE:.*]] = %[[ACC_INIT:.*]]) -> (token, token) {
// The store is ordered after the wait, but through the loop-invariant root
// token rather than the loop-carried lastOp block argument.
// CHECK:   %[[STORE_INPUT:.*]] = join_tokens %[[ROOT]], %[[WAIT]] : token
// CHECK:   %[[STORE:.*]] = store_view_tko weak {{.*}}[%[[IV]], {{.*}}] token{{ ?}}={{ ?}}%[[STORE_INPUT]]
// CHECK:   %[[JOIN:.*]] = join_tokens %[[LASTSTORE]], %[[STORE]] : token
// CHECK:   continue %[[JOIN]], %[[JOIN]] : token, token
// CHECK: }
// Dependent launch consumes producer writes and the acquire token.
// CHECK: %[[LAUNCH_INPUT:.*]] = join_tokens %[[FOR]]#1, %[[WAIT]] : token
// CHECK: gdc_launch_dependents_tko token{{ ?}}={{ ?}}%[[LAUNCH_INPUT]] -> token

// -----

// Test: same GDC wait/launch envelope, but the loop stores the same tile every
// iteration. There is no disjoint proof, so the loop-carried WAW edge through
// lastOp must be preserved.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_gdc_wait_constant_index_store_loop_before_launch(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64x64xf32>
      %tview = cuda_tile.make_tensor_view %arg0, shape=[256, 256], strides=[256, 1] : tensor_view<256x256xf32, strides=[256,1]>
      %pview = cuda_tile.make_partition_view %tview : partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>
      %wait = cuda_tile.gdc_wait_tko -> token
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %store = cuda_tile.store_view_tko weak %tile, %pview[%cst_0_i32, %cst_0_i32] : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> token
        cuda_tile.continue
      }
      %launch = cuda_tile.gdc_launch_dependents_tko -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_gdc_wait_constant_index_store_loop_before_launch
// CHECK: %[[WAIT:.*]] = gdc_wait_tko{{.*}}-> token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %{{.*}}, {{.*}}) -> (token, token) {
// The constant-index store may alias earlier iterations, so it must consume the
// loop-carried lastOp token in addition to the wait token.
// CHECK:   %[[STORE_INPUT:.*]] = join_tokens %[[LASTOP]], %[[WAIT]] : token
// CHECK:   %[[STORE:.*]] = store_view_tko weak {{.*}} token{{ ?}}={{ ?}}%[[STORE_INPUT]]
// CHECK:   continue %[[STORE]], %[[STORE]] : token, token
// CHECK: }
// CHECK: %[[LAUNCH_INPUT:.*]] = join_tokens %[[FOR]]#1, %[[WAIT]] : token
// CHECK: gdc_launch_dependents_tko token{{ ?}}={{ ?}}%[[LAUNCH_INPUT]] -> token
