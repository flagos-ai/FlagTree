// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s --implicit-check-not="flag = -1" --implicit-check-not="flag = 15" --implicit-check-not="<PIPE_FIX>, <PIPE_V>] flag = 2" --implicit-check-not="<PIPE_MTE3>, <PIPE_MTE1>] flag = 1"

// Regression for the cross-direction flag-id reuse bug. A Vector->Cube copy
// (set[VECTOR, MTE3, MTE1]) used to be merged onto the same flag as an adjacent
// Cube->Vector fixpipe wait (wait[VECTOR, FIX, V]), e.g. in the wild:
//
//     hivm.hir.sync_block_set [<VECTOR>, <PIPE_MTE3>, <PIPE_MTE1>] flag = 1
//     hivm.hir.sync_block_wait[<VECTOR>, <PIPE_FIX>,  <PIPE_V>]    flag = 1   <- WRONG
//
// The two directions run concurrently on the two cores, so a shared counting
// flag lets the wait steal the set's signal. They must never share a flag id.
// Here a long Vector->Cube->Vector matmul chain needs more than MAX_FLAG_ID
// flags, forcing reuse; the result must partition strictly by direction.

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @flag_reuse_cross_direction() {
    %cst = arith.constant {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} 0.000000e+00 : f32
    %e0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>
    %v0 = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%e0 : tensor<32x32xf32>) -> tensor<32x32xf32>

    // round 1: V->C (matmul consumes %v0), then C->V (exp consumes %m1)
    %ef1 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %m1 = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%v0, %v0 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ef1 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %v1 = math.exp %m1 {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    %ef2 = tensor.empty() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %m2 = linalg.matmul {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "CUBE"} ins(%v1, %v1 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ef2 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %v2 = math.exp %m2 {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    %ef3 = tensor.empty() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %m3 = linalg.matmul {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} ins(%v2, %v2 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ef3 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %v3 = math.exp %m3 {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    %ef4 = tensor.empty() {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %m4 = linalg.matmul {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "CUBE"} ins(%v3, %v3 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ef4 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %v4 = math.exp %m4 {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    %ef5 = tensor.empty() {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %m5 = linalg.matmul {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "CUBE"} ins(%v4, %v4 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ef5 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %v5 = math.exp %m5 {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    %ef6 = tensor.empty() {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %m6 = linalg.matmul {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} ins(%v5, %v5 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ef6 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %v6 = math.exp %m6 {ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    %ef7 = tensor.empty() {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %m7 = linalg.matmul {ssbuffer.block_id = 14 : i32, ssbuffer.core_type = "CUBE"} ins(%v6, %v6 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ef7 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %v7 = math.exp %m7 {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    %ef8 = tensor.empty() {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : tensor<32x32xf32>
    %m8 = linalg.matmul {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} ins(%v7, %v7 : tensor<32x32xf32>, tensor<32x32xf32>) outs(%ef8 : tensor<32x32xf32>) -> tensor<32x32xf32>
    %v8 = math.exp %m8 {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "VECTOR"} : tensor<32x32xf32>

    return
  }
}

// CHECK-LABEL: func.func @flag_reuse_cross_direction
// Vector->Cube (MTE3/MTE1) copies all reuse one flag; Cube->Vector (FIX/V)
// fixpipes all reuse another; the two never collide.
// CHECK: <PIPE_MTE3>, <PIPE_MTE1>] flag = 2
// CHECK: <PIPE_FIX>, <PIPE_V>] flag = 1
// CHECK: return
