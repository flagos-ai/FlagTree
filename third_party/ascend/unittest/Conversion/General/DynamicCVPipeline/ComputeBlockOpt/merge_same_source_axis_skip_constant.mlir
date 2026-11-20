// RUN: triton-opt --merge-same-source-axis %s | FileCheck %s

// Guards: arith.constant must NOT act as a merge source.
//
// %c0_i32 (block 31) is consumed by %79 and %80 (both block 27). %80 is
// also a direct user of %79. Without the constant-source guard the BFS
// would trigger a trivial 1-step convergence at %80 and reblock both ops
// to block 31 — which is exactly the leak pattern that motivated this
// guard. With the guard, the constant is skipped and nothing moves.

// CHECK-LABEL: func.func @constant_source_skipped
func.func @constant_source_skipped(%arg0: i32, %arg1: i32) {
  // CHECK: arith.constant {{.*}}ssbuffer.block_id = 31
  %c0_i32 = arith.constant {MixUse, ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32
  // CHECK: arith.constant {{.*}}ssbuffer.block_id = 31
  %c1_i32 = arith.constant {MixUse, ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} 1 : i32

  // These two must stay at block 27 — constant is not a real axis source.
  // CHECK: arith.subi {{.*}}ssbuffer.block_id = 27
  %79 = arith.subi %c0_i32, %arg0 {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "VECTOR"} : i32
  // CHECK: arith.maxsi {{.*}}ssbuffer.block_id = 27
  %80 = arith.maxsi %79, %c0_i32 {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "VECTOR"} : i32
  // CHECK: arith.index_cast {{.*}}ssbuffer.block_id = 27
  %81 = arith.index_cast %80 {ssbuffer.block_id = 27 : i32, ssbuffer.core_type = "VECTOR"} : i32 to index

  return
}

// Constants used purely as scalar operands to a tensor-typed merge chain
// must also not perturb the chain. Here %src is a real tensor source at
// block 28; the chain below is a legitimate same-source-different-axes
// pattern and must still merge.
func.func @constant_consumer_does_not_break_legit_chain(%arg0: tensor<8xf32>, %arg1: f32) {
  %c0_i32 = arith.constant {ssbuffer.block_id = 31 : i32, ssbuffer.core_type = "VECTOR"} 0 : i32

  // Source %src at block 28 — tensor so the pass treats it as a candidate.
  %src = arith.addf %arg0, %arg0 {ssbuffer.block_id = 28 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  // Two consumers with a shared constant operand — they must still merge.
  // CHECK: arith.mulf {{.*}}ssbuffer.block_id = 28
  %a = arith.mulf %src, %arg0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.addf {{.*}}ssbuffer.block_id = 28
  %b = arith.addf %src, %arg0 {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>
  // CHECK: arith.subf {{.*}}ssbuffer.block_id = 28
  %k = arith.subf %a, %b {ssbuffer.block_id = 29 : i32, ssbuffer.core_type = "VECTOR"} : tensor<8xf32>

  return
}
