// RUN: triton-opt %s --flagtree-insert-debug-records | FileCheck %s

module attributes {flagtree.debug.record_level = 1 : i32} {
  // CHECK-LABEL: func.func @doc_example
  // CHECK-SAME: flagtree.debug.hidden_arg = "__debug_ctrl_ptr"
  // CHECK-SAME: flagtree.debug.logical_instance_id_formula
  func.func @doc_example(%x: f32, %a: f32, %b: f32) {
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()

    // CHECK: %[[Y:.*]] = arith.mulf
    // CHECK-SAME: flagtree.debug.instrumented = true
    // CHECK-SAME: flagtree.debug.record_kinds = ["summary"]
    // CHECK-SAME: flagtree.debug.triton_statement = "y = tl.dot(a, b)"
    %y = arith.mulf %a, %b {
      flagtree.debug.op_id = 1 : i32,
      flagtree.debug.scope_id = 1 : i32,
      flagtree.debug.triton_statement = "y = tl.dot(a, b)"
    } : f32
    // CHECK: "flagtree_debug.record_summary_bundle"(%[[Y]])
    // CHECK-SAME: op_id = 1 : i32
    // CHECK-SAME: scope_id = 1 : i32

    // CHECK: %[[Z:.*]] = arith.addf
    // CHECK-SAME: flagtree.debug.instrumented = true
    // CHECK-SAME: flagtree.debug.record_kinds = ["summary"]
    // CHECK-SAME: flagtree.debug.triton_statement = "z = x + y"
    %z = arith.addf %x, %y {
      flagtree.debug.op_id = 2 : i32,
      flagtree.debug.scope_id = 1 : i32,
      flagtree.debug.triton_statement = "z = x + y"
    } : f32
    // CHECK: "flagtree_debug.record_summary_bundle"(%[[Z]])
    // CHECK-SAME: op_id = 2 : i32
    // CHECK-SAME: scope_id = 1 : i32

    "flagtree_debug.collect_end"() : () -> ()
    func.return
  }
}
