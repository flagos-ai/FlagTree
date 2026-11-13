// RUN: triton-opt --merge-vector-if-block %s | FileCheck %s

// A pure-Vector scf.if whose only data source (condition + values read inside)
// is block_id 10. The if (block_id 16) and its inner op (block_id 8) must be
// merged into the upstream block_id 10, and the first downstream consumer block
// (block_id 11) must be merged into 10 as well, while the second downstream
// block (block_id 12) is left untouched.
module {
  // CHECK-LABEL: func.func @merge_pure_vector_if
  func.func @merge_pure_vector_if(%arg0: f32, %arg1: f32) {
    %one = arith.constant {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} 1.000000e+00 : f32
    %eps = arith.constant {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} 9.99999997E-7 : f32

    // upstream data source: block_id 10
    %ext = arith.addf %arg0, %arg1 {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : f32
    %cond = arith.cmpf ogt, %ext, %eps {ssbuffer.block_id = 10 : i32, ssbuffer.core_type = "VECTOR"} : f32

    // pure-Vector if: outer block_id 16, inner block_id 8
    // CHECK: scf.if
    // CHECK: arith.divf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 10
    // CHECK: } {ssbuffer.block_id = 10
    %r = scf.if %cond -> (f32) {
      %d = arith.divf %one, %ext {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : f32
      scf.yield {ssbuffer.core_type = "VECTOR"} %d : f32
    } else {
      scf.yield {ssbuffer.core_type = "VECTOR"} %one : f32
    } {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "VECTOR"}

    // first downstream consumer: block_id 11 -> merged into 10
    // CHECK: arith.mulf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 10
    %u1 = arith.mulf %r, %ext {ssbuffer.block_id = 11 : i32, ssbuffer.core_type = "VECTOR"} : f32

    // second downstream consumer: block_id 12 -> unchanged
    // CHECK: arith.addf %{{.*}}, %{{.*}} {{{.*}}ssbuffer.block_id = 12
    %u2 = arith.addf %r, %arg0 {ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "VECTOR"} : f32
    return
  }
}
