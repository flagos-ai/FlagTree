// RUN: triton-opt --merge-same-source-axis %s | FileCheck %s

// Guards: if the convergence op has an upstream operand whose defOp lives
// in a block that this merge will NOT cover (neither source's block, nor
// any block of an op in chainOps), the merge must be skipped. Otherwise
// moving K to srcBlockId would leave a cross-block dependency between the
// moved K and the unaligned defOp.

// CHECK-LABEL: func.func @convergence_with_external_dep
func.func @convergence_with_external_dep(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) {
  // External tensor at block 50 — will NOT be moved by this merge.
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 50
  %ext = arith.addf %arg0, %arg1 {ssbuffer.block_id = 50 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  // Source at block 28.
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 28
  %src = arith.addf %arg0, %arg1 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  // Two consumers at block 29 — convergent chain.
  // CHECK: arith.mulf {{.*}}ssbuffer.block_id = 29
  %a = arith.mulf %src, %arg0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 29
  %b = arith.addf %src, %arg1 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  // Convergence uses both consumers AND %ext. The guard must skip the
  // merge — moving %k to block 28 while %ext stays at block 50 would
  // create a cross-block dependency that ReorderOpsByBlockId cannot
  // safely resolve.
  // CHECK: arith.subf {{.*}}ssbuffer.block_id = 29
  %k = arith.subf %a, %ext {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  return
}

// Even when the upstream defOp IS already in srcBlockId (28), the merge
// is still rejected. reason: willCreateCycle's DFS treats `%ext(28) →
// %k(29→28)` as an internal cycle within the target block, so the merge
// is skipped. This is the correct conservative behavior — making the
// chain self-referential at the target block (an upstream op at the
// target block feeding a chain op that just arrived at the target block)
// is exactly what the cycle detector is meant to prevent.
func.func @external_dep_in_src_block_ok(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) {
  // Upstream tensor already at srcBlockId.
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 28
  %ext = arith.addf %arg0, %arg1 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 28
  %src = arith.addf %arg0, %arg1 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  // Chain stays at block 29 — willCreateCycle rejects the merge.
  // CHECK: arith.mulf {{.*}}ssbuffer.block_id = 29
  %a = arith.mulf %src, %arg0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 29
  %b = arith.addf %src, %arg1 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.subf {{.*}}ssbuffer.block_id = 29
  %k = arith.subf %a, %ext {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  return
}

// Upstream defOp that is itself part of chainOps is fine too: it will be
// moved along with K, so no cross-block dep survives the merge.
func.func @upstream_defop_already_in_chain(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) {
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 28
  %src = arith.addf %arg0, %arg1 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  // %a and %b both consume %src. Their convergent chain ends at %k,
  // but %b is also an operand of %k — i.e. %b's defOp is in chainOps.
  // CHECK: arith.mulf {{.*}}ssbuffer.block_id = 28
  %a = arith.mulf %src, %arg0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.mulf {{.*}}ssbuffer.block_id = 28
  %b = arith.mulf %src, %arg1 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.subf {{.*}}ssbuffer.block_id = 28
  %k = arith.subf %a, %b {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  return
}
