// RUN: triton-opt --merge-same-source-axis %s | FileCheck %s

// Source %src is in block_id 28. Two consumers in block_id 29 do different
// "axis" processing of %src (here: mulf by different scalars) and converge
// at %sub = arith.subf %b1, %b2 (also block_id 29). All three should be
// moved to block_id 28.
//
// %src also has unrelated consumers in block 30 and block 31 that must NOT
// be touched.
module {
  // CHECK-LABEL: func.func @same_source_diff_axis
  func.func @same_source_diff_axis(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) {
    // Source in block 28.
    %src = arith.addf %arg0, %arg1 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

    // Convergent chain in block 29 — all three should move to block 28.
    // CHECK: arith.mulf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 28
    %b1 = arith.mulf %src, %arg0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
    // CHECK: arith.mulf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 28
    %b2 = arith.mulf %src, %arg1 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
    // CHECK: arith.subf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 28
    %sub = arith.subf %b1, %b2 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

    // Unrelated single consumer in block 30 — must stay.
    // CHECK: arith.addf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 30
    %u1 = arith.addf %src, %arg0 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

    // Unrelated single consumer in block 31 — must stay.
    // CHECK: arith.mulf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 31
    %u2 = arith.mulf %src, %arg1 {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

    return
  }
}

// CHECK-LABEL: func.func @same_source_diff_axis_cross_block
// Consumers u1 (block 29) and u2 (block 30) live in DIFFERENT blocks, but
// converge at %sub (block 30). All three should still move to source's
// block_id (50).
func.func @same_source_diff_axis_cross_block(%arg0: tensor<8xf32>, %arg1: tensor<8xf32>) {
  // Source in block 50.
  // CHECK: arith.addf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 50
  %src = arith.addf %arg0, %arg1 {ssbuffer.block_id = 50 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  // u1 in block 29, u2 in block 30 — different blocks, but the chain still
  // converges at %sub.
  // CHECK: arith.mulf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 50
  %u1 = arith.mulf %src, %arg0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.mulf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 50
  %u2 = arith.mulf %src, %arg1 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.subf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 50
  %sub = arith.subf %u1, %u2 {ssbuffer.block_id = 30 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  return
}
