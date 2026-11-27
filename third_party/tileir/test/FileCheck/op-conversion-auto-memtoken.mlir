// RUN: triton-cuda-tile-opt %s -split-input-file --pass-pipeline="builtin.module(convert-triton-to-cuda-tile,cuda_tile.module(cuda_tile.entry(fuse-fma)),reconcile-unrealized-casts,cuda_tile.module(cuda_tile.entry(auto-gen-memory-token{autogen-alias-memtoken=true})))" | FileCheck %s

// Keep this file focused on conversion-pipeline smoke tests. Detailed
// AutoGenMemoryToken ordering semantics are covered by auto-memtoken-*.mlir.

// Test: nested scf.if with mixed loads/stores on distinct/overlapping
// classes, followed by post-if load + store on the same class (Ret).
// The outer if yields token(s) for every (class, role) advanced in either
// branch. Post-if ops consume the merged token via the if results.
module {
  tt.func public @test_auto_memtoken_if_normal(%Cond: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %XTrue: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %XFalse: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %Ret: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c2_i32 = arith.constant 2 : i32
    %pid = tt.get_program_id x : i32
    %0 = arith.remsi %pid, %c2_i32 : i32
    %1 = arith.cmpi eq, %0, %c0_i32 : i32
    scf.if %1 {
      %2 = tt.load %XTrue : !tt.ptr<f32>
      tt.store %Ret, %2 : !tt.ptr<f32>
    } else {
      %2 = arith.cmpi eq, %0, %c1_i32 : i32
      scf.if %2 {
        %3 = tt.load %XFalse : !tt.ptr<f32>
        tt.store %Ret, %3 : !tt.ptr<f32>
      }
    }
    %4 = tt.load %Ret : !tt.ptr<f32>
    %5 = arith.addf %4, %4 : f32
    tt.store %Ret, %5 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_if_normal
// The outer scf.if yields token results for every (class, role) advanced
// in either branch — at least the Ret class's lastStore, which the post-
// if load consumes as its RAW input.
// CHECK: %[[IFRES:.*]]:{{[0-9]+}} = if {{.*}} -> (token,
// CHECK:   store_ptr_tko
// CHECK:   yield
// CHECK: } else {
// CHECK:   yield
// CHECK: }
// Post-if load on Ret: takes one of the if's token results as RAW input.
// CHECK: {{.*}}, %[[POST_LOAD_TOK:.*]] = load_ptr_tko {{.*}} token=%[[IFRES]]#{{[0-9]+}}
// Post-if store on Ret: WAR+WAW — joins prior lastOp (from if) with the
// read's output before writing.
// CHECK: join_tokens
// CHECK: store_ptr_tko {{.*}} token=


// -----

// Test: scf.for with load+store on the same ptr class. Body threads
// lastOp+lastStore via iter_args; each iter WAR-joins the iter_arg with
// the read's output before storing.
module {
  tt.func public @test_auto_memtoken_for(%Out1: !tt.ptr<i64>, %Out2: !tt.ptr<i64>) attributes {noinline = false} {
    %c10000_i32 = arith.constant 10000 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<1> : tensor<128xi64>
    %c1_i32 = arith.constant 1 : i32
    %off = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %0 = tt.splat %Out1 : !tt.ptr<i64> -> tensor<128x!tt.ptr<i64>>
    %1 = tt.addptr %0, %off : tensor<128x!tt.ptr<i64>>, tensor<128xi32>
    scf.for %i = %c0_i32 to %c10000_i32 step %c1_i32  : i32 {
      %3 = tt.load %1 : tensor<128x!tt.ptr<i64>>
      %4 = arith.addi %3, %cst : tensor<128xi64>
      tt.store %1, %4 : tensor<128x!tt.ptr<i64>>
    }
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_for
// make_token init for the Out1 class's lastOp + lastStore (2 tokens).
// CHECK: make_token : token
// CHECK: make_token : token
// Loop threads 2 token iter_args; body uses iter_arg for RAW load input,
// then JOINs with the read's output as WAR input for the store, then
// continues with the store's output as both new lastOp and new lastStore.
// CHECK: for {{.*}} iter_values({{.*}} = %{{.*}}, {{.*}} = %{{.*}}) -> (token, token) {
// CHECK:   {{.*}}, %[[LOAD_TOK:.*]] = load_ptr_tko {{.*}} token=
// CHECK:   %[[JOIN:.*]] = join_tokens
// CHECK:   %[[STORE_TOK:.*]] = store_ptr_tko {{.*}} token=%[[JOIN]]
// CHECK:   continue %[[STORE_TOK]], %[[STORE_TOK]] : token, token
// CHECK: }


// -----

// Test: scf.for containing an scf.while whose body does load+store.
// Outer for threads tokens for the data class; inner while does the same
// and yields results back out.
module {
  tt.func public @test_auto_memtoken_nested_while(%data: !tt.ptr<f32>, %countPtr: !tt.ptr<i32>) attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    scf.for %i = %c0_i32 to %c10_i32 step %c1_i32  : i32 {
      %count = tt.load %countPtr : !tt.ptr<i32>
      %count_0 = scf.while (%count_1 = %count) : (i32) -> i32 {
        %0 = arith.cmpi sgt, %count_1, %c0_i32 : i32
        scf.condition(%0) %count_1 : i32
      } do {
      ^bb0(%count_1: i32):
        %0 = tt.load %data : !tt.ptr<f32>
        %1 = arith.addf %0, %cst : f32
        tt.store %data, %1 : !tt.ptr<f32>
        %count_2 = arith.subi %count_1, %c2_i32 : i32
        scf.yield %count_2 : i32
      }
    }
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_nested_while
// CHECK: for {{.*}} iter_values(
// CHECK:   load_ptr_tko
// CHECK:   loop iter_values(
// CHECK:     if {{.*}} {
// CHECK:       load_ptr_tko
// CHECK:       join_tokens
// CHECK:       store_ptr_tko
// CHECK:       continue
// CHECK:     } else {
// CHECK:       break
// CHECK:     }
// CHECK:   }
// CHECK:   continue
// CHECK: }


// -----

// Test: GDC ops participate in token ordering. gdc_wait installs its output
// as acquireToken; every subsequent memory op JOINs with acquireToken.
// gdc_launch_dependents is a dependent-launch signal: it is anchored after
// prior producer writes/acquireToken, but does not order later same-kernel
// memory or unrelated read-only state.
module {
  tt.func public @test_auto_memtoken_gdc_fence(%In: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %Out: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %dummy_wait = tt.elementwise_inline_asm "griddepcontrol.wait; // dummy $0" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32
    %0 = tt.load %In : !tt.ptr<f32>
    tt.store %Out, %0 : !tt.ptr<f32>
    %1 = tt.load %Out : !tt.ptr<f32>
    %2 = arith.addf %1, %1 : f32
    tt.store %Out, %2 : !tt.ptr<f32>
    %dummy_launch = tt.elementwise_inline_asm "griddepcontrol.launch_dependents; // dummy $0" {constraints = "=r", packed_element = 1 : i32, pure = false} -> i32
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_gdc_fence
// CHECK: %[[GDC_WAIT:.*]] = gdc_wait_tko {{.*}}-> token
// Load_In uses acquireToken directly as RAW input (lastStore[In] is null).
// CHECK: load_ptr_tko {{.*}} token=%[[GDC_WAIT]]
// First Store_Out also uses acquireToken (lastOp[Out] is null).
// CHECK: store_ptr_tko {{.*}} token=%[[GDC_WAIT]]
// Load_Out (RAW on Out): chains to the store + acquireToken.
// CHECK: load_ptr_tko {{.*}} token=
// Second Store_Out WARs against load_out + WAWs against store_out + acquireToken.
// CHECK: join_tokens
// CHECK: store_ptr_tko {{.*}} token=
// Dependent launch is anchored after prior producer writes/acquireToken.
// CHECK: join_tokens
// CHECK: gdc_launch_dependents_tko

// -----

// Test: two scf.for loops access the same ptr root via distinct SSA
// induction variables — loop 1 stores, loop 2 loads. Under the
// alias-class keying, both loops land in the same class, so the pass
// chains store → load via a token iter_arg on the read loop.
module {
  tt.func public @test_auto_memtoken_cross_loop_same_base(%Out: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %cst = arith.constant 1.000000e+00 : f32
    %c0_f32 = arith.constant 0.000000e+00 : f32
    scf.for %i = %c0_i32 to %c4_i32 step %c1_i32  : i32 {
      %p = tt.addptr %Out, %i : !tt.ptr<f32>, i32
      tt.store %p, %cst : !tt.ptr<f32>
    }
    %acc = scf.for %j = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%a = %c0_f32) -> (f32) : i32 {
      %p = tt.addptr %Out, %j : !tt.ptr<f32>, i32
      %v = tt.load %p : !tt.ptr<f32>
      %sum = arith.addf %a, %v : f32
      scf.yield %sum : f32
    }
    tt.store %Out, %acc : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_cross_loop_same_base
// Loop 1: store-only body threads lastOp + lastStore as 2 iter_args.
// CHECK: %[[STORE_LOOP:.*]]:2 = for {{.*}} iter_values({{.*}} = %{{.*}}, {{.*}} = %{{.*}}) -> (token, token) {
// CHECK:   %[[STORE_TOK:.*]] = store_ptr_tko {{.*}} token=
// CHECK:   %[[STORE_JOIN:.*]] = join_tokens {{.*}}, %[[STORE_TOK]]
// CHECK:   continue %[[STORE_JOIN]], %[[STORE_JOIN]]
// CHECK: }
// Loop 2: load-only body. Body effect is Load, so only lastOp is threaded
// (1 token iter_arg). The load consumes loop 1's lastStore result directly
// (%[[STORE_LOOP]]#1) as a loop-invariant RAW input.
// CHECK: for {{.*}} iter_values({{.*}}, {{.*}} = %[[STORE_LOOP]]#0) -> ({{.*}}, token) {
// CHECK:   load_ptr_tko {{.*}} token=%[[STORE_LOOP]]#1
// CHECK:   continue
// CHECK: }
// Post-loop store on the same class chains after loop 2's lastOp result.
// CHECK: store_ptr_tko {{.*}} token=

// -----

// Test: pointer descriptor carried through scf.for iter_args — the load
// inside the loop uses the iter_arg as its descriptor, and a pre-loop
// store on the same base must be RAW-ordered with it. Requires the alias
// walk to recurse through cuda_tile.for block-arg sources (init + every
// cuda_tile.continue operand) instead of falling to a terminal root.
module {
  tt.func public @test_auto_memtoken_for_iter_ptr(%Base: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c4 = arith.constant 4 : i32
    %cst = arith.constant 1.000000e+00 : f32
    // Pre-loop store on %Base.
    tt.store %Base, %cst : !tt.ptr<f32>
    // Loop that carries the pointer itself via iter_args. Each iteration
    // loads through the iter_arg (not through %Base directly) and advances
    // the pointer for the next iteration.
    %final = scf.for %i = %c0 to %c4 step %c1 iter_args(%p = %Base) -> (!tt.ptr<f32>) : i32 {
      %v = tt.load %p : !tt.ptr<f32>
      tt.store %p, %v : !tt.ptr<f32>
      %next = tt.addptr %p, %c1 : !tt.ptr<f32>, i32
      scf.yield %next : !tt.ptr<f32>
    }
    tt.return
  }
}

// CHECK-LABEL: @test_auto_memtoken_for_iter_ptr
// Pre-loop store on %Base produces a store token. The alias walk must
// unify this class with the for-loop's pointer iter_arg class (by
// recursing through init + continue operand) so the class already has
// lastOp/lastStore populated when the loop is processed.
// CHECK: %[[PRE_STORE:.*]] = store_ptr_tko weak %{{.*}}, %{{.*}} : tile<ptr<f32>>, tile<f32> -> token
// Loop iter_values carry the pre-loop store token into both lastOp and
// lastStore slots (no fresh make_token).
// CHECK: for {{.*}} iter_values(%{{.*}} = %{{.*}}, %[[ARG1:.*]] = %[[PRE_STORE]], %[[ARG2:.*]] = %[[PRE_STORE]]) -> (tile<ptr<f32>>, token, token) {
// In-body load consumes the lastStore iter_arg as RAW input.
// CHECK:   %{{.*}}, %[[LOAD_TOK:.*]] = load_ptr_tko {{.*}} token=%[[ARG2]]
// In-body store consumes the eager-join of lastOp + this iter's load
// token as WAR input.
// CHECK:   %[[JOIN:.*]] = join_tokens %[[ARG1]], %[[LOAD_TOK]]
// CHECK:   %[[ST_TOK:.*]] = store_ptr_tko {{.*}} token=%[[JOIN]]
// Continue yields store token into both token iter_args.
// CHECK:   continue %{{.*}}, %[[ST_TOK]], %[[ST_TOK]] : tile<ptr<f32>>, token, token
