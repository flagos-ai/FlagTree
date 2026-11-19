// RUN: triton-opt --merge-same-source-axis %s | FileCheck %s

// Guards: 1-step "false convergence" must be rejected.
//
// Two VECTOR_ONLY consumers of a tensor source are in a direct
// producer-consumer relationship (%A -> %B). Without the trivial-
// convergence guard, the BFS would step once from %A and encounter %B,
// declare %B the convergence op, and reblock the chain — even though
// there is no real two-branch merge.
//
// With the guard, hitting another starting consumer is treated as a
// non-event and the BFS keeps exploring. No downstream op is reached by
// two independent paths, so findNearestConvergence returns false.

// CHECK-LABEL: func.func @no_false_convergence_via_producer_consumer
func.func @no_false_convergence_via_producer_consumer(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) {
  // Tensor source at block 28 (will be tried as a merge root).
  %src = arith.addf %arg0, %arg1 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  // %a is a direct consumer of %src (block 27). It has only one
  // downstream user %b, which is also a consumer of %src (block 27).
  // The BFS would naively see %a -> %b as a "convergence at %b" but %b
  // is itself a starting consumer — must be rejected.
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 27
  %a = arith.addf %src, %arg1 {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.mulf {{.*}}ssbuffer.block_id = 27
  %b = arith.mulf %a, %arg1 {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  return
}

// Same pattern but the "trivial convergence" candidate is a deeper
// consumer that the BFS would also reach in one step. Still rejected.
func.func @no_false_convergence_via_chained_consumer(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) {
  %src = arith.addf %arg0, %arg1 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 30
  %x = arith.addf %src, %arg1 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.mulf {{.*}}ssbuffer.block_id = 30
  %y = arith.mulf %x, %arg1 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // %z is downstream of both %x and %y paths only via %y; not a real
  // two-branch merge. BFS from %x reaches %y in one step, but %y is a
  // starting consumer — must be rejected.
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 31
  %z = arith.addf %y, %arg0 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  return
}
