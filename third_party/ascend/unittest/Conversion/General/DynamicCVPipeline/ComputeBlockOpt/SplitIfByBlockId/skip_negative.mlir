// RUN: triton-opt --split-input-file --split-if-by-block-id %s | FileCheck %s

// VCV ordering WITHOUT cross-core data consumption: %m consumes no VECTOR
// value and no VECTOR op consumes %m. Should NOT split even though the
// group order is VECTOR -> CUBE -> VECTOR.
// CHECK-LABEL: func.func @skip_vcv_no_data_flow
// CHECK-COUNT-1: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 96
func.func @skip_vcv_no_data_flow(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: f32, %d: f32, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : f32
    scf.if %cond {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : f32
      %m = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v2 = arith.mulf %c, %d {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : f32
    }
  }
  return
}

// -----

// Non-CVC/VCV: else side has only 2 groups (CUBE+VECTOR), not alternating 3 groups, should NOT split.
// CHECK-LABEL: func.func @skip_two_groups
// CHECK-COUNT-1: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
func.func @skip_two_groups(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: f32, %d: f32, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : f32
    scf.if %cond {
      %x = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : f32
    } else {
      %y = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
    }
  }
  return
}

// -----

// Non-CVC/VCV: else side has 2 pure VECTOR groups (no matmul), then side has
// 1 group. shouldSplitThen/shouldSplitElse are both false while elseGroups
// has >= 2 entries; without the needsSplit() guard this would read
// ya.groupOutputs out of bounds in materializeCandidate.
// CHECK-LABEL: func.func @skip_else_pure_vector
// CHECK-COUNT-1: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 95
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 96
func.func @skip_else_pure_vector(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: f32, %d: f32, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : f32
    scf.if %cond {
      %x = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : f32
    } else {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "VECTOR"} : f32
      %v2 = arith.mulf %c, %d {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : f32
    }
  }
  return
}

// -----

// ForOp with both CUBE and VECTOR ops inside: getCoreTypeOfSimpleOpOrCf
// returns CUBE_AND_VECTOR, which groupIsCube treats as unclassified.
// With no CUBE group, should NOT split.
// CHECK-LABEL: func.func @skip_vcv_for_mixed
// CHECK-COUNT-1: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// CHECK: scf.for
// CHECK: linalg.matmul {{.*}}CUBE
// CHECK: arith.addf {{.*}}VECTOR
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 96
func.func @skip_vcv_for_mixed(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    scf.if %cond {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %for_res = scf.for %i = %c0 to %c1 step %c1 iter_args(%arg = %v1) -> (tensor<2x2xf32>) {
        %m = linalg.matmul ins(%arg, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
        %inner = arith.addf %m, %m {ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
        scf.yield %inner : tensor<2x2xf32>
      } {ssbuffer.block_id = 95 : i32}
      %v2 = arith.mulf %for_res, %b {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    }
  }
  return
}

// -----

// Group with CUBE_AND_VECTOR op: not CUBE_ONLY, not VECTOR_ONLY,
// groupIsCube treats it as unclassified. With no CUBE group, should NOT split.
// CHECK-LABEL: func.func @skip_cube_and_vector_group
// CHECK-COUNT-1: scf.if
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 94
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 95
// CHECK: arith.mulf {{.*}}ssbuffer.block_id = 96
func.func @skip_cube_and_vector_group(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    scf.if %cond {
      %v1 = arith.addf %c, %d {ssbuffer.block_id = 94 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
      %v2 = arith.addf %v1, %c {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE_AND_VECTOR"} : tensor<2x2xf32>
      %v3 = arith.mulf %v2, %b {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    }
  }
  return
}

// -----

// CCV: then side has CUBE -> CUBE -> VECTOR, no trailing CUBE group.
// Data flow is only CUBE->VECTOR (no VECTOR->CUBE), so should NOT split.
// CHECK-LABEL: func.func @skip_ccv_no_trailing_cube
// CHECK-COUNT-1: scf.if
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 95
// CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 96
// CHECK: arith.addf {{.*}}ssbuffer.block_id = 97
func.func @skip_ccv_no_trailing_cube(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>, %d: tensor<2x2xf32>, %cond: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %iv = %c0 to %c1 step %c1 {
    %cube = arith.addf %c, %d {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "CUBE"} : tensor<2x2xf32>
    scf.if %cond {
      %m1 = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 95 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %m2 = linalg.matmul ins(%m1, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%a : tensor<2x2xf32>) {ssbuffer.block_id = 96 : i32, ssbuffer.core_type = "CUBE"} -> tensor<2x2xf32>
      %v = arith.addf %m1, %m2 {ssbuffer.block_id = 97 : i32, ssbuffer.core_type = "VECTOR"} : tensor<2x2xf32>
    }
  }
  return
}
