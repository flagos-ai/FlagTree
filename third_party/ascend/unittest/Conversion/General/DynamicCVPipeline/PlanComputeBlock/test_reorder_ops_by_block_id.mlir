// RUN: triton-opt --reorder-ops-by-block-id %s | FileCheck %s

// Tests for orderInOneCBlock in ReorderOpsByBlockIdPass.
//
// The pass reorders operations within a compute block using a Kahn's
// topological sort with a tie-breaker that sinks store-like ops
// (materialize_in_destination / hivm.hir.store) to the end of VECTOR_ONLY
// blocks, while respecting SSA and memory dependencies.
// CUBE blocks (any block containing a non-VECTOR_ONLY op) preserve the
// original program order.

module {
  // ============================================================================
  // 1. cube_block_store_not_last
  //
  // A CUBE block where bufferization.materialize_in_destination is NOT the
  // last op. In CUBE blocks, orderInOneCBlock preserves the original program
  // order (store-like ops are NOT sunk).
  //
  // Program order:  materialize, matmul
  // Expected:       materialize, matmul  (unchanged)
  // ============================================================================
  // CHECK-LABEL: func.func @cube_block_store_not_last(
  // CHECK: bufferization.materialize_in_destination {{.*}} ssbuffer.core_type = "CUBE"
  // CHECK: linalg.matmul {{.*}} ssbuffer.core_type = "CUBE"
  func.func @cube_block_store_not_last(
      %arg0: memref<64x64xf16>,
      %in1: tensor<64x64xf16>,
      %A: tensor<64x64xf16>,
      %B: tensor<64x64xf16>,
      %init: tensor<64x64xf16>) {
    bufferization.materialize_in_destination %in1 in writable %arg0 {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : (tensor<64x64xf16>, memref<64x64xf16>) -> ()
    %mm = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
      ins(%A, %B : tensor<64x64xf16>, tensor<64x64xf16>)
      outs(%init : tensor<64x64xf16>) -> tensor<64x64xf16>
    return
  }

  // ============================================================================
  // 2. vector_materialize_in_middle
  //
  // A VECTOR_ONLY block where bufferization.materialize_in_destination sits
  // between two independent VECTOR compute ops. After reorder, the store
  // sinks past the independent compute op to the end.
  //
  // Program order:  addf, materialize(addf), mulf
  // Expected:       addf, mulf, materialize
  // ============================================================================
  // CHECK-LABEL: func.func @vector_materialize_in_middle(
  // CHECK: arith.addf {{.*}} ssbuffer.core_type = "VECTOR"
  // CHECK: arith.mulf {{.*}} ssbuffer.core_type = "VECTOR"
  // CHECK: bufferization.materialize_in_destination {{.*}} ssbuffer.core_type = "VECTOR"
  func.func @vector_materialize_in_middle(
      %arg0: memref<64xf32>,
      %in1: tensor<64xf32>,
      %in2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
    %a = arith.addf %in1, %in1 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    bufferization.materialize_in_destination %a in writable %arg0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64xf32>, memref<64xf32>) -> ()
    %b = arith.mulf %in2, %in2 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    return %a, %b : tensor<64xf32>, tensor<64xf32>
  }

  // ============================================================================
  // 3. vector_store_inside_scf_if_sinks
  //
  // bufferization.materialize_in_destination is wrapped inside an scf.if.
  // The scf.if carries ssbuffer.block_id and ssbuffer.core_type so it is in
  // the same VECTOR_ONLY compute block. The scf.if returns a value used by
  // another VECTOR compute op. Since the scf.if contains a store-like op,
  // isStoreLikeWithRegion(scf.if) == true, so the scf.if itself is treated
  // as store-like and sinks past independent VECTOR ops, while respecting
  // the SSA dependency from the consuming compute op.
  //
  // Program order:  scf.if(+store), independent_mulf, use_addf(if_result)
  // Expected:       independent_mulf, scf.if, use_addf
  //                  (scf.if sinks past mulf but not past use_addf which
  //                   depends on it)
  // ============================================================================
  // CHECK-LABEL: func.func @vector_store_inside_scf_if_sinks(
  // CHECK: arith.mulf {{.*}} ssbuffer.core_type = "VECTOR"
  // CHECK: scf.if
  // CHECK: } {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: arith.addf {{.*}} ssbuffer.core_type = "VECTOR"
  func.func @vector_store_inside_scf_if_sinks(
      %cond: i1,
      %arg0: memref<64xf32>,
      %in: tensor<64xf32>,
      %in2: tensor<64xf32>) -> (tensor<64xf32>, tensor<64xf32>) {
    %result = "scf.if"(%cond) ({
      %val = arith.addf %in, %in {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      bufferization.materialize_in_destination %val in writable %arg0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64xf32>, memref<64xf32>) -> ()
      "scf.yield"(%val) : (tensor<64xf32>) -> ()
    }, {
      %val2 = arith.mulf %in, %in {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
      bufferization.materialize_in_destination %val2 in writable %arg0 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (tensor<64xf32>, memref<64xf32>) -> ()
      "scf.yield"(%val2) : (tensor<64xf32>) -> ()
    }) {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : (i1) -> tensor<64xf32>
    %independent = arith.mulf %in2, %in2 {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    %use = arith.addf %result, %result {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<64xf32>
    return %independent, %use : tensor<64xf32>, tensor<64xf32>
  }
}
