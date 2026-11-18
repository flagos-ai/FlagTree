// RUN: triton-opt --merge-vector-if-block %s | FileCheck %s

// Cycle-detection guard for the shared willCreateCycle helper used by
// merge-vector-if-block.
//
// Merging the pure-Vector scf.if together with its downstream block_id 11 would
// create a block-level cycle: block_id 13 reads the upstream (block_id 10) and
// also feeds block_id 11, so unifying the if + block_id 11 into block_id 10
// closes the loop 10 -> 13 -> 11(=10). The pass must therefore SKIP the
// downstream merge and fall back to folding the if into the upstream block_id 10
// only, leaving block_id 13 and block_id 11 untouched.
//
// NOTE: block_id is a module-global namespace in ComputeBlockIdManager, so this
// case lives in its own file to avoid cross-function block_id interference.
module {
  // CHECK-LABEL: func.func @merge_if_downstream_cycle
  func.func @merge_if_downstream_cycle(%arg0: f32, %arg1: f32) {
    %eps = arith.constant {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} 9.99999997E-7 : f32

    // upstream data source: block_id 10
    %ext = arith.addf %arg0, %arg1 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %cond = arith.cmpf ogt, %ext, %eps {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : f32

    // block_id 13 consumes upstream (%ext) and feeds the downstream block_id 11.
    // It must stay at block_id 13.
    // CHECK: %[[T1:.*]] = arith.mulf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 13
    %t1 = arith.mulf %ext, %arg0 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : f32
    // CHECK: %[[T2:.*]] = arith.addf %[[T1]], %{{.*}} {{{.*}}ssbuffer.block_id = 13
    %t2 = arith.addf %t1, %arg1 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : f32

    // pure-Vector if (outer block_id 16, inner block_id 8); its only data source
    // is block_id 10, so the fallback folds it (and its inner op) into block_id 10.
    // CHECK: scf.if
    // CHECK: arith.divf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 10
    // CHECK: } {ssbuffer.block_id = 10
    %r = scf.if %cond -> (f32) {
      %d = arith.divf %ext, %arg0 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : f32
      scf.yield {ssbuffer.core_type = "VECTOR"} %d : f32
    } else {
      scf.yield {ssbuffer.core_type = "VECTOR"} %ext : f32
    } {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"}

    // downstream block_id 11 must stay untouched (merging it would create a cycle).
    // CHECK: arith.mulf %{{.*}}, %[[T2]] {{{.*}}ssbuffer.block_id = 11
    %u1 = arith.mulf %r, %t2 {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : f32
    return
  }
}
