// RUN: triton-opt %s --flagtree-resolve-debug-scope --flagtree-assign-debug-op-id --flagtree-insert-debug-records | FileCheck %s

module attributes {flagtree.debug.addr_level = 1 : i32, flagtree.debug.record_level = 1 : i32} {
  // CHECK-LABEL: func.func @pipeline
  // CHECK-SAME: flagtree.debug.hidden_arg = "__debug_ctrl_ptr"
  func.func @pipeline(%arg0: memref<1xf32>, %idx: index) {
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
    // CHECK: %[[LOAD:.*]] = memref.load
    // CHECK-SAME: flagtree.debug.instrumented = true
    // CHECK-SAME: flagtree.debug.memory_event_kind = "LAST_ALIGNED_ADDR"
    // CHECK-SAME: flagtree.debug.op_id = 1 : i32
    // CHECK-SAME: flagtree.debug.record_kinds = ["summary", "memory_event"]
    %0 = memref.load %arg0[%idx] : memref<1xf32>
    // CHECK: "flagtree_debug.record_summary_bundle"(%[[LOAD]])
    // CHECK-SAME: op_id = 1 : i32
    // CHECK-SAME: scope_id = 1 : i32
    // CHECK: "flagtree_debug.capture_memory_address"(%arg0)
    // CHECK-SAME: op_id = 1 : i32
    // CHECK-SAME: scope_id = 1 : i32

    // CHECK: %[[ADD:.*]] = arith.addf
    // CHECK-SAME: flagtree.debug.instrumented = true
    // CHECK-SAME: flagtree.debug.op_id = 2 : i32
    // CHECK-SAME: flagtree.debug.record_kinds = ["summary"]
    %1 = arith.addf %0, %0 : f32
    // CHECK: "flagtree_debug.record_summary_bundle"(%[[ADD]])
    // CHECK-SAME: op_id = 2 : i32
    // CHECK-SAME: scope_id = 1 : i32
    "flagtree_debug.collect_end"() : () -> ()
    func.return
  }
}
