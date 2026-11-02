// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(cuda_tile.module(cuda_tile.entry(auto-gen-memory-token{autogen-alias-memtoken=true})))" 2>/dev/null | FileCheck %s

// Test: cuda_tile.for body with a nested cuda_tile.if whose branches
// both end in cuda_tile.continue. handleForOp must forward termOps
// into addMemTokenForBlock so handleIfOp can collect the nested
// continues — otherwise the loop rewrite grows the iter_arg signature
// but the nested continues keep their original 0-operand count,
// triggering a verifier failure.
//
// Hand-written cuda_tile IR — no Triton frontend produces this pattern
// because scf.for + scf.if's scf.yield converts to cuda_tile.yield, not
// cuda_tile.continue. Input is already post-conversion cuda_tile, so the
// pass-pipeline omits convert-triton-to-cuda-tile and runs the token pass
// in isolation.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_for_nested_continue(%arg0: tile<ptr<f32>>, %arg1: tile<i1>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %cst_0_f32 = cuda_tile.constant <f32: 0.000000e+00> : tile<f32>
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        cuda_tile.if %arg1 {
          %t_then = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
          cuda_tile.continue
        } else {
          %t_else = cuda_tile.store_ptr_tko weak %arg0, %cst_0_f32 : tile<ptr<f32>>, tile<f32> -> token
          cuda_tile.continue
        }
      }
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_for_nested_continue
// The for loop threads lastOp+lastStore iter_args for %arg0's class (2).
// CHECK: for {{.*}} iter_values(%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}}) -> (token, token)
// Both nested cuda_tile.continue terminators must carry two token
// operands matching the rewritten loop iter_arg count.
// CHECK:   if {{.*}} {
// CHECK:     %[[THEN_TOK:.*]] = store_ptr_tko
// CHECK:     continue %[[THEN_TOK]], %[[THEN_TOK]] : token, token
// CHECK:   } else {
// CHECK:     %[[ELSE_TOK:.*]] = store_ptr_tko
// CHECK:     continue %[[ELSE_TOK]], %[[ELSE_TOK]] : token, token
// CHECK:   }

// -----

// Test: cuda_tile.return as a terminator of a branch inside a
// cuda_tile.if. ReturnOp's ParentOneOf includes IfOp, so a branch can
// exit via return without yielding to the if. handleIfOp must treat
// return branches as non-yielding — otherwise the pass appends token
// operands to cuda_tile.return (ill-formed: return doesn't yield to the
// if) and rebuilds the if expecting both branches to produce matching
// result values.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_if_return_branch(%arg0: tile<ptr<f32>>, %arg1: tile<i1>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      cuda_tile.if %arg1 {
        %t_then = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
        cuda_tile.return
      } else {
        %t_else = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
        cuda_tile.yield
      }
      %t_post = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_if_return_branch
// The if is rebuilt with token results from the (yielding) else branch.
// The then branch keeps its original cuda_tile.return terminator — no
// token operands appended (that would be invalid, return doesn't yield
// to the if).
// CHECK: if %{{.*}} -> (token, token)
// CHECK:   %[[T_THEN:.*]] = store_ptr_tko
// CHECK:   return
// CHECK: } else {
// CHECK:   %[[T_ELSE:.*]] = store_ptr_tko
// CHECK:   yield %[[T_ELSE]], %[[T_ELSE]] : token, token
// CHECK: }
// Post-if store chains after the if's yielded lastStore token (from the
// else branch's yield, the only reachable exit).
// CHECK: store_ptr_tko {{.*}} token=

// -----

// Test: manual memToken + gpu.barrier → barrier must be erased, not
// left in the IR. The manual-token bail path emits a warning saying
// the barrier will be ignored; the pass must actually remove the
// barrier to match. Downstream cuda_tile lowering expects this pass
// to consume debug barriers.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_manual_token_erases_barrier(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %tok = cuda_tile.make_token : token
      %0 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 token=%tok : tile<ptr<f32>>, tile<f32> -> token
      gpu.barrier
      %1 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 token=%0 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_manual_token_erases_barrier
// CHECK: make_token
// CHECK: store_ptr_tko
// Barrier must be gone — the pass bails on manual tokens but must still
// erase gpu.barrier, matching the emitted warning.
// CHECK-NOT: gpu.barrier
// CHECK: store_ptr_tko
// CHECK: return

// -----

// Test: mixed cuda_tile.if inside cuda_tile.for — then branch ends in
// cuda_tile.continue (early-skip), else branch ends in cuda_tile.yield
// (fall-through). handleIfOp must classify each branch terminator
// independently: collect only loop-exit branches into termOps, and
// fold only yielding branches into the post-if state. A blanket
// decision based on the then terminator alone would either push the
// else's yield into termOps (and the enclosing loop would extend it
// with loop-iter-arg-sized operands, breaking the yield's contract
// with the if's 0 results) or send the else's continue through the
// yield rewrite.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_for_if_continue_yield_mix(%arg0: tile<ptr<f32>>, %arg1: tile<i1>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %cst_0_f32 = cuda_tile.constant <f32: 0.000000e+00> : tile<f32>
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        cuda_tile.if %arg1 {
          %t_then = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
          cuda_tile.continue
        } else {
          %t_else = cuda_tile.store_ptr_tko weak %arg0, %cst_0_f32 : tile<ptr<f32>>, tile<f32> -> token
          cuda_tile.yield
        }
        cuda_tile.continue
      }
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_for_if_continue_yield_mix
// Loop threads lastOp + lastStore for %arg0 (2 token iter_args).
// CHECK: for {{.*}} iter_values({{.*}}) -> (token, token)
// The if is rebuilt with 2 token results — the else branch yields
// them (it flows through); the then branch is a loop-exit so it's NOT
// extended with if-result-sized operands. Instead the then's continue
// is pushed into the loop's termOps and carries LOOP-iter-arg-sized
// operands (2 tokens, same count as the loop's threads).
// CHECK:   if %{{.*}} -> (token, token) {
// CHECK:     %[[THEN_TOK:.*]] = store_ptr_tko
// CHECK:     continue %[[THEN_TOK]], %[[THEN_TOK]] : token, token
// CHECK:   } else {
// CHECK:     %[[ELSE_TOK:.*]] = store_ptr_tko
// CHECK:     yield %[[ELSE_TOK]], %[[ELSE_TOK]] : token, token
// CHECK:   }
// Main body tail continue uses the if's yielded results (reached only
// when the else branch ran).
// CHECK:   continue %{{.*}}#0, %{{.*}}#1 : token, token

// -----

// Test: a cuda_tile.store_ptr_tko placed after
// cuda_tile.gdc_launch_dependents_tko. The launch is anchored after the
// pre-launch store, but its output must not be broadcast into later unrelated
// memory ops. Post-launch work is allowed to overlap the dependent kernel's
// pre-wait work.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_op_after_launch_signal(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %s0 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      %launch = cuda_tile.gdc_launch_dependents_tko -> token
      %s1 = cuda_tile.store_ptr_tko weak %arg1, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_op_after_launch_signal
// The pre-launch store runs first.
// CHECK: %[[S0:.*]] = store_ptr_tko
// The launch consumes the pre-launch store's token.
// CHECK: %[[LAUNCH:.*]] = {{.*}}gdc_launch_dependents_tko token = %[[S0]] -> token
// The post-launch store on a different class is independent of the launch.
// CHECK-NEXT: store_ptr_tko weak %{{.*}}, %{{.*}} : tile<ptr<f32>>, tile<f32> -> token

// -----

// Test: advance markers must survive across a nested control-flow
// handler. Pattern: the then branch of an outer if does `store A;
// cuda_tile.for { ... }` where the loop does not touch A. If
// handleForOp clears state.advanced* on entry, the branch-local
// marker that A's lastStore was advanced is destroyed; the outer
// handleIfOp's union then misses A and the if is rebuilt without an
// A token result, so a post-if store on A has no RAW dependency on
// the conditional store inside the branch.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_marker_survives_nested_for(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>, %cond: tile<i1>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      cuda_tile.if %cond {
        // Branch-local advance: lastStore[A] = %tA.
        %tA = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
        // Nested loop that only touches %arg1 (different class).
        cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
          %tB = cuda_tile.store_ptr_tko weak %arg1, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
          cuda_tile.continue
        }
        cuda_tile.yield
      } else {
        cuda_tile.yield
      }
      // Post-if store on A. If the if is rebuilt without an A token
      // result, this store has no RAW dependency on the branch-local
      // store and can be reordered ahead of it.
      %tA2 = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_marker_survives_nested_for
// The if is rebuilt with a token result for A — the then branch
// advanced A's lastStore even though the nested for didn't. The
// post-if store on A consumes the if's yielded token, so it
// happens-after the conditional store.
// CHECK: %[[IF:.*]]{{:[0-9]+|#[0-9]+}} = if {{.*}} -> ({{.*}}token{{.*}}) {
// CHECK: store_ptr_tko {{.*}} token=%[[IF]]#{{[0-9]+}}

// -----

// Test: a gpu.barrier inside a cuda_tile.for, plus a post-loop mem op
// on an alias class that the loop body does NOT touch. RegionInfo
// must track total fences: the loop handler threads acquireToken and
// on exit propagates its final value into lastOp for every unthreaded
// class. Otherwise the barrier's in-loop fan-out is lost at loop exit
// and post-loop ops on untouched classes get no dependency on the
// barrier, letting the scheduler reorder them ahead of it.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_barrier_in_loop_propagates(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      // Pre-loop store to give %arg0 class a defined lastOp.
      %pre = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        // Loop body writes %arg1 (different class) and emits a
        // gpu.barrier — the barrier broadcasts to every class
        // (including %arg0, which is otherwise untouched inside the
        // loop).
        %t1 = cuda_tile.store_ptr_tko weak %arg1, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
        gpu.barrier
        cuda_tile.continue
      }
      // Post-loop load on %arg0. If the barrier's broadcast doesn't
      // escape the loop, the load's input token is just the pre-loop
      // store and the barrier is invisible to it.
      %post, %post_tok = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_barrier_in_loop_propagates
// Loop threads acquireToken (presence of gpu.barrier triggers the
// hasReleaseFence path — gpu.barrier is acq+rel).
// CHECK: for {{.*}} iter_values({{.*}}) -> ({{.*}}token{{.*}}) {
// CHECK:   store_ptr_tko
// Inside the loop body, the barrier is consumed by the pass (erased)
// and its semantics are carried via the acquireToken iter_arg.
// CHECK-NOT:   gpu.barrier
// Post-loop load on %arg0 consumes a token — crucially, not nothing,
// and not just the pre-loop store token. The in-loop barrier reaches
// it through the acquireToken propagation.
// CHECK: load_ptr_tko {{.*}} token=

// -----

// Test: a cuda_tile.for that performs one weak store_view_tko per
// iteration to a partition_view indexed by the loop induction variable
// should not serialize the stores through the loop-carried lastOp
// token. Each store depends on the loop-invariant root token, while
// the loop accumulates all store result tokens through join_tokens so
// the post-loop load still RAW-depends on every store.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_parallel_store_view_loop(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64x64xf32>
      %tview = cuda_tile.make_tensor_view %arg0, shape=[256, 256], strides=[256, 1] : tensor_view<256x256xf32, strides=[256,1]>
      %pview = cuda_tile.make_partition_view %tview : partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %store = cuda_tile.store_view_tko weak %tile, %pview[%i, %cst_0_i32] : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_view_tko weak %pview[%cst_0_i32, %cst_0_i32] : partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> tile<64x64xf32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_parallel_store_view_loop
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[ACC_INIT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT]], %[[LASTSTORE:.*]] = %[[ACC_INIT]]) -> (token, token) {
// CHECK:   %[[STORE:.*]] = store_view_tko weak {{.*}}[%[[IV]], {{.*}}] token{{ ?}}={{ ?}}%[[ROOT]]
// CHECK:   %[[JOIN:.*]] = join_tokens %[[LASTSTORE]], %[[STORE]] : token
// CHECK:   continue %[[JOIN]], %[[JOIN]] : token, token
// CHECK: }
// CHECK: load_view_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: a no-memory nested if/continue before the parallel store must not
// turn the store's loop-invariant root token into a loop-carried token. The
// early-continue path yields the incoming loop-carried lastOp/lastStore
// because no store ran on that path; the fall-through path still issues the
// store with ROOT and accumulates only the actual store result into lastStore.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_parallel_store_view_loop_with_skip(%arg0: tile<ptr<f32>>, %cond: tile<i1>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64x64xf32>
      %tview = cuda_tile.make_tensor_view %arg0, shape=[256, 256], strides=[256, 1] : tensor_view<256x256xf32, strides=[256,1]>
      %pview = cuda_tile.make_partition_view %tview : partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        cuda_tile.if %cond {
          cuda_tile.continue
        } else {
          cuda_tile.yield
        }
        %store = cuda_tile.store_view_tko weak %tile, %pview[%i, %cst_0_i32] : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_view_tko weak %pview[%cst_0_i32, %cst_0_i32] : partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> tile<64x64xf32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_parallel_store_view_loop_with_skip
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[ACC_INIT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT]], %[[LASTSTORE:.*]] = %[[ACC_INIT]]) -> (token, token) {
// CHECK:   if {{.*}} {
// CHECK-NEXT:     continue %[[LASTOP]], %[[LASTSTORE]] : token, token
// CHECK-NEXT:   } else {
// CHECK-NEXT:   }
// CHECK:   %[[STORE:.*]] = store_view_tko weak {{.*}}[%[[IV]], {{.*}}] token{{ ?}}={{ ?}}%[[ROOT]]
// CHECK:   %[[JOIN:.*]] = join_tokens %[[LASTSTORE]], %[[STORE]] : token
// CHECK:   continue %[[JOIN]], %[[JOIN]] : token, token
// CHECK: }
// CHECK: load_view_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: the parallel-store shortcut is not applied when the store
// index is loop-invariant. Those iterations may overwrite the same
// tile, so the original WAW chain through the loop-carried lastOp
// token must be preserved.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_constant_index_store_view_loop(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %tile = cuda_tile.constant <f32: 1.000000e+00> : tile<64x64xf32>
      %tview = cuda_tile.make_tensor_view %arg0, shape=[256, 256], strides=[256, 1] : tensor_view<256x256xf32, strides=[256,1]>
      %pview = cuda_tile.make_partition_view %tview : partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %store = cuda_tile.store_view_tko weak %tile, %pview[%cst_0_i32, %cst_0_i32] : tile<64x64xf32>, partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> token
        cuda_tile.continue
      }
      %loaded, %load_token = cuda_tile.load_view_tko weak %pview[%cst_0_i32, %cst_0_i32] : partition_view<tile=(64x64), tensor_view<256x256xf32, strides=[256,1]>>, tile<i32> -> tile<64x64xf32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_constant_index_store_view_loop
// CHECK: %[[ROOT:.*]] = make_token
// CHECK: %[[ACC_INIT:.*]] = make_token
// CHECK: %[[FOR:.*]]:2 = for %[[IV:.*]] in {{.*}} iter_values(%[[LASTOP:.*]] = %[[ROOT]], %[[LASTSTORE:.*]] = %[[ACC_INIT]]) -> (token, token) {
// CHECK:   %[[STORE:.*]] = store_view_tko weak {{.*}} token{{ ?}}={{ ?}}%[[LASTOP]]
// CHECK:   continue %[[STORE]], %[[STORE]] : token, token
// CHECK: }
// CHECK: load_view_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]#1

// -----

// Test: a load-only cuda_tile.for whose alias class has writes ONLY before
// (not after) the loop in the function should NOT thread its lastOp out as
// an iter_arg — the previous-iter load tokens have no future write to
// WAR-against, so the iter_arg is dead and blocks the pipeliner.
//
// Position-sensitive downstream-effects analysis (handleForOp) decides
// LOAD-only threading from "writes/fences AFTER this loop" rather than
// the function-global writeCountByClass.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_load_only_loop_no_post_write(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      // Pre-loop write on %arg0 (writeCountByClass[%arg0] > 0 globally).
      %pre = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      // Loop body has only loads on %arg0 — no body-internal write.
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %v, %t = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
        cuda_tile.continue
      }
      // Post-loop op on %arg1 (different class) — NO downstream write or
      // fence on %arg0's class. The load-only loop should not thread.
      %post = cuda_tile.store_ptr_tko weak %arg1, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_load_only_loop_no_post_write
// Body has only loads on %arg0; no post-loop write/fence on %arg0's class.
// Loop must run with NO token iter_args (no `iter_values` clause and no
// arrow-result type — both would appear immediately before the `{` if a
// token were threaded).
// CHECK: for %{{.*}} in {{.*}} : tile<i32> {
// CHECK:   load_ptr_tko weak %{{.*}} token{{ ?}}={{ ?}}%{{.*}}
// In-loop load still RAWs against the pre-loop store via the outer SSA
// value (no cross-iter chain needed for a load-only class).

// -----

// Test: a load-only cuda_tile.for followed by an acquire-only fence does not
// thread lastOp out of the loop. gdc_wait_tko publishes a token for later
// memory ops, but it does not consume prior local memory state.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_load_only_loop_before_wait(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %v, %t = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
        cuda_tile.continue
      }
      %wait = cuda_tile.gdc_wait_tko -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_load_only_loop_before_wait
// CHECK: for %{{.*}} in {{.*}} : tile<i32> {
// CHECK:   load_ptr_tko weak {{.*}}
// CHECK-NOT: join_tokens
// CHECK: }
// CHECK: %[[WAIT:.*]] = gdc_wait_tko -> token

// -----

// Test: a load-only cuda_tile.for followed by a dependent-launch signal does
// not thread lastOp out of the loop. The signal consumes producer writes, not
// read-only local memory state.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_load_only_loop_before_launch(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %v, %t = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
        cuda_tile.continue
      }
      %launch = cuda_tile.gdc_launch_dependents_tko -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_load_only_loop_before_launch
// CHECK: for %{{.*}} in {{.*}} : tile<i32> {
// CHECK:   load_ptr_tko weak {{.*}}
// CHECK-NOT: join_tokens
// CHECK: }
// CHECK: %[[LAUNCH:.*]] = gdc_launch_dependents_tko -> token

// -----

// Test: same shape as above but WITH a post-loop write on %arg0's class.
// handleForOp must thread lastOp[c_arg0] so the post-loop write WAR-
// orders against the in-loop loads.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_load_only_loop_with_post_write(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %pre = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.for %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %v, %t = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
        cuda_tile.continue
      }
      // Post-loop write on the SAME class — load-only loop must thread
      // lastOp[c_arg0] so this write WARs against the in-loop loads.
      %post = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_load_only_loop_with_post_write
// Loop threads ONE token iter_arg (lastOp accumulator for %arg0).
// CHECK: %[[FOR:.*]] = for %{{.*}} in {{.*}} : tile<i32> iter_values(%[[ARG:.*]] = %{{.*}}) -> (token) {
// CHECK:   %{{.*}}, %[[LOAD_TOK:.*]] = load_ptr_tko weak {{.*}}
// CHECK:   %[[JOIN:.*]] = join_tokens %[[ARG]], %[[LOAD_TOK]]
// CHECK:   continue %[[JOIN]] : token
// CHECK: }
// Post-loop write WAR-orders against loop's accumulated lastOp.
// CHECK: store_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%[[FOR]]

// -----

// Test: rewriting a cuda_tile.for to thread tokens must preserve the unsigned
// comparison flag and discardable attributes on the original loop.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_preserve_unsigned_for(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_0_i32 = cuda_tile.constant <i32: 0> : tile<i32>
      %cst_1_i32 = cuda_tile.constant <i32: 1> : tile<i32>
      %cst_4_i32 = cuda_tile.constant <i32: 4> : tile<i32>
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      cuda_tile.for unsigned %i in (%cst_0_i32 to %cst_4_i32, step %cst_1_i32) : tile<i32> {
        %store = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
        cuda_tile.continue
      } {memtoken_test = "keep"}
      %loaded, %load_token = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_preserve_unsigned_for
// CHECK: for unsigned %{{.*}} in {{.*}} : tile<i32> iter_values({{.*}}) -> (token, token) {
// CHECK: } {memtoken_test = "keep"}
// CHECK: load_ptr_tko weak {{.*}} token{{ ?}}={{ ?}}%{{.*}}

// -----

// Test: direct return from a cuda_tile.loop exits the enclosing function; it
// must not be extended with loop token operands when the loop is rebuilt.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_loop_direct_return(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      cuda_tile.loop {
        %store = cuda_tile.store_ptr_tko weak %arg0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
        %wait = cuda_tile.gdc_wait_tko -> token
        cuda_tile.return
      }
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_loop_direct_return
// CHECK: loop iter_values({{.*}}) : token, token, token -> token, token, token {
// CHECK:   store_ptr_tko weak
// CHECK:   gdc_wait_tko
// CHECK-NEXT:   return{{$}}

// -----

// Test: same acquire-only fence behavior as above, but for cuda_tile.loop.
// gdc_wait_tko does not consume prior local memory state, so a preceding
// load-only loop does not need to thread lastOp.

module {
  cuda_tile.module @cuda_tile_module {
    entry @test_auto_memtoken_load_only_cuda_loop_before_wait(%arg0: tile<ptr<f32>>) optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      cuda_tile.loop {
        %v, %t = cuda_tile.load_ptr_tko weak %arg0 : tile<ptr<f32>> -> tile<f32>, token
        cuda_tile.break
      }
      %wait = cuda_tile.gdc_wait_tko -> token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_load_only_cuda_loop_before_wait
// CHECK: loop {
// CHECK:   load_ptr_tko weak {{.*}}
// CHECK-NOT: join_tokens
// CHECK:   break
// CHECK: }
// CHECK: %[[WAIT:.*]] = gdc_wait_tko -> token

// -----

// Test: repeated get_global ops for the same symbol name the same mutable
// allocation, so they must share an alias root.

module {
  cuda_tile.module @cuda_tile_module {
    cuda_tile.global @g <f32: [0.000000e+00]> : tile<1xf32>
    entry @test_auto_memtoken_get_global_alias_root() optimization_hints=<sm_100 = {num_cta_in_cga = 1, num_worker_warps_per_cta = 4, occupancy = 1}> {
      %p0 = cuda_tile.get_global @g : tile<ptr<f32>>
      %p1 = cuda_tile.get_global @g : tile<ptr<f32>>
      %cst_1_f32 = cuda_tile.constant <f32: 1.000000e+00> : tile<f32>
      %store = cuda_tile.store_ptr_tko weak %p0, %cst_1_f32 : tile<ptr<f32>>, tile<f32> -> token
      %loaded, %load_token = cuda_tile.load_ptr_tko weak %p1 : tile<ptr<f32>> -> tile<f32>, token
      cuda_tile.return
    }
  }
}

// CHECK-LABEL: @test_auto_memtoken_get_global_alias_root
// CHECK: %[[P0:.*]] = get_global @g : tile<ptr<f32>>
// CHECK: %[[P1:.*]] = get_global @g : tile<ptr<f32>>
// CHECK: %[[STORE:.*]] = store_ptr_tko weak %[[P0]]
// CHECK: load_ptr_tko weak %[[P1]] token{{ ?}}={{ ?}}%[[STORE]]
