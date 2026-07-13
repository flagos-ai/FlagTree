// RUN: triton-opt %s --flagtree-insert-debug-records | FileCheck %s

module attributes {flagtree.debug.addr_level = 1 : i32, flagtree.debug.record_level = 1 : i32} {
  // CHECK-LABEL: func.func @memref_copy_addresses
  // CHECK-SAME: (%[[SRC:[a-zA-Z0-9_]+]]: memref<16xf32>, %[[DST:[a-zA-Z0-9_]+]]: memref<16xf32>)
  // CHECK-SAME: flagtree.debug.hidden_arg = "__debug_ctrl_ptr"
  func.func @memref_copy_addresses(%src: memref<16xf32>, %dst: memref<16xf32>) {
    // CHECK: memref.copy %[[SRC]], %[[DST]]
    // CHECK-SAME: flagtree.debug.instrumented = true
    // CHECK-SAME: flagtree.debug.memory_event_kind = "LAST_ALIGNED_ADDR"
    // CHECK-SAME: flagtree.debug.record_kinds = ["memory_event"]
    memref.copy %src, %dst {flagtree.debug.op_id = 12 : i32, flagtree.debug.is_memory_op = true} : memref<16xf32> to memref<16xf32>
    // CHECK: "flagtree_debug.capture_memory_address"(%[[SRC]])
    // CHECK-SAME: op_id = 12 : i32
    // CHECK-SAME: operand_index = 0 : i32
    // CHECK: "flagtree_debug.capture_memory_address"(%[[DST]])
    // CHECK-SAME: op_id = 12 : i32
    // CHECK-SAME: operand_index = 1 : i32
    func.return
  }
}
