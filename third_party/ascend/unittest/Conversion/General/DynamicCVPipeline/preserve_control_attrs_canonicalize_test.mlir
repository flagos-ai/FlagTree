// RUN: triton-opt %s --preserve-control-attrs-canonicalize --mlir-print-op-generic | FileCheck %s

// CHECK-LABEL: "func.func"() <{{.*}}sym_name = "for_attr_survives_unused_iter_arg_canonicalize"
// CHECK: "scf.for"
// CHECK: }) {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 1 : i32

module {
  func.func @for_attr_survives_unused_iter_arg_canonicalize(%arg0: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %0:2 = scf.for %iv = %c0 to %c4 step %c1 iter_args(%acc = %arg0, %dead = %c0_i32) -> (i32, i32) {
      %1 = arith.addi %acc, %c1_i32 : i32
      scf.yield %1, %dead : i32, i32
    } {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 1 : i32}
    return %0#0 : i32
  }
}

// -----

// CHECK-LABEL: "func.func"() <{{.*}}sym_name = "while_attr_survives_unused_result_canonicalize"
// CHECK: "scf.while"
// CHECK: }) {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 1 : i32} : (i32, i32) -> (i32, i32)

module {
  func.func @while_attr_survives_unused_result_canonicalize(%arg0: i32, %limit: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0:2 = scf.while (%acc = %arg0, %dead = %c0_i32) : (i32, i32) -> (i32, i32) {
      %1 = arith.cmpi slt, %dead, %limit : i32
      scf.condition(%1) %acc, %dead : i32, i32
    } do {
    ^bb0(%acc_iter: i32, %dead_iter: i32):
      %1 = arith.addi %acc_iter, %c1_i32 : i32
      %2 = arith.addi %dead_iter, %c1_i32 : i32
      scf.yield %1, %2 : i32, i32
    } attributes {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 1 : i32}
    return %0#0 : i32
  }
}

// -----

// CHECK-LABEL: "func.func"() <{{.*}}sym_name = "if_attr_survives_unused_result_canonicalize"
// CHECK-DAG: "arith.select"
// CHECK-DAG: "scf.if"
// CHECK: }) {hivm.unlikely_condition, ssbuffer.block_id = 11 : i32} : (i1) -> ()

module {
  func.func @if_attr_survives_unused_result_canonicalize(%cond: i1, %arg0: i32, %buf: memref<1xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0:2 = "scf.if"(%cond) ({
      memref.store %arg0, %buf[%c0] : memref<1xi32>
      "scf.yield"(%arg0, %c0_i32) : (i32, i32) -> ()
    }, {
      memref.store %c0_i32, %buf[%c0] : memref<1xi32>
      "scf.yield"(%c0_i32, %arg0) : (i32, i32) -> ()
    }) {hivm.unlikely_condition, ssbuffer.block_id = 11 : i32} : (i1) -> (i32, i32)
    return %0#0 : i32
  }
}

// -----

// CHECK-LABEL: "func.func"() <{{.*}}sym_name = "merged_zero_result_if_preserves_attrs"
// CHECK: "scf.if"
// CHECK: "memref.store"
// CHECK: "memref.store"
// CHECK: }) {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32} : (i1) -> ()
// CHECK-NOT: "scf.if"
// CHECK: "func.return"

module {
  func.func @merged_zero_result_if_preserves_attrs(%cond: i1, %lhs: memref<1xi32>, %rhs: memref<1xi32>) {
    %c0 = arith.constant 0 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    scf.if %cond {
      memref.store %c1_i32, %lhs[%c0] : memref<1xi32>
    } {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32}
    scf.if %cond {
      memref.store %c2_i32, %rhs[%c0] : memref<1xi32>
    } {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32}
    return
  }
}


// -----

// CHECK-LABEL: "func.func"() <{{.*}}sym_name = "nested_for_replacement_keeps_outer_main_loop"
// CHECK: %{{.*}} = "scf.for"(
// CHECK: }) {ssbuffer.block_id = 16 : i32} : (index, index, index, i32) -> i32
// CHECK-NOT: ssbuffer.main_loop
// CHECK-NOT: ssbuffer.core_type
// CHECK-NEXT: "scf.yield"
// CHECK-NEXT: }) {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 0 : i32} : (index, index, index, i32) -> i32

module {
  func.func @nested_for_replacement_keeps_outer_main_loop(%seed: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0:2 = scf.for %iv = %c0 to %c2 step %c1 iter_args(%acc = %seed, %dead = %c0_i32) -> (i32, i32) {
      %1 = scf.for %j = %c0 to %c2 step %c1 iter_args(%sum = %acc) -> (i32) {
        %2 = arith.addi %sum, %c1_i32 : i32
        scf.yield %2 : i32
      } {ssbuffer.block_id = 16 : i32}
      scf.yield %1, %dead : i32, i32
    } {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "VECTOR", ssbuffer.main_loop = 0 : i32}
    return %0#0 : i32
  }
}

// -----


// CHECK-LABEL: "func.func"() <{{.*}}sym_name = "nested_multi_for_main_loop_not_inner"
// CHECK: %{{.*}}:2 = "scf.for"(
// CHECK: }) {ssbuffer.block_id = 16 : i32} : (index, index, index, i32) -> i32
// CHECK-NOT: ssbuffer.main_loop
// CHECK-NOT: ssbuffer.pipe_stage
// CHECK-NEXT: %{{.*}} = "scf.for"(
// CHECK-NEXT: ^bb0
// CHECK: }) {ssbuffer.pipe_stage = 3 : i32} : (index, index, index, i32) -> i32
// CHECK-NOT: ssbuffer.main_loop
// CHECK-NOT: ssbuffer.block_id
// CHECK-NEXT: "scf.yield"
// CHECK-NEXT: }) {ssbuffer.main_loop = 0 : i32} : (index, index, index, i32, i32) -> (i32, i32)
// CHECK-NOT: ssbuffer.block_id
// CHECK-NOT: ssbuffer.pipe_stage

module {
  func.func @nested_multi_for_main_loop_not_inner(%seed: i32) -> (i32, i32) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %0:2 = scf.for %iv = %c0 to %c2 step %c1 iter_args(%acc = %seed, %dead = %c0_i32) -> (i32, i32) {
      %1 = scf.for %j = %c0 to %c2 step %c1 iter_args(%sum = %acc) -> (i32) {
        %2 = arith.addi %sum, %c1_i32 : i32
        scf.yield %2 : i32
      } {ssbuffer.block_id = 16 : i32}
      %3 = scf.for %k = %c0 to %c2 step %c1 iter_args(%sum2 = %acc) -> (i32) {
        %4 = arith.addi %sum2, %c1_i32 : i32
        scf.yield %4 : i32
      } {ssbuffer.pipe_stage = 3 : i32}
      scf.yield %1, %3 : i32, i32
    } {ssbuffer.main_loop = 0 : i32}
    return %0#0, %0#1 : i32, i32
  }
}
