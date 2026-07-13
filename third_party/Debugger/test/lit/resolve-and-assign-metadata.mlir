// RUN: triton-opt %s -split-input-file --flagtree-resolve-debug-scope --flagtree-assign-debug-op-id | FileCheck %s

module {
  func.func @metadata(%arg0: memref<4xf32>, %idx: index) {
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
    %0 = memref.load %arg0[%idx] : memref<4xf32>
    %1 = arith.addf %0, %0 : f32
    memref.store %1, %arg0[%idx] : memref<4xf32>
    "flagtree_debug.collect_end"() : () -> ()
    %2 = arith.addf %0, %0 : f32
    func.return
  }
}

// CHECK-LABEL: func.func @metadata
// CHECK-NOT: flagtree_debug.collect_begin
// CHECK-NOT: flagtree_debug.collect_end
// CHECK: memref.load
// CHECK-SAME: flagtree.debug.addr_level = 0 : i32
// CHECK-SAME: flagtree.debug.is_memory_op = true
// CHECK-SAME: flagtree.debug.op_category = "load"
// CHECK-SAME: flagtree.debug.op_id = 1 : i32
// CHECK-SAME: flagtree.debug.record_level = 1 : i32
// CHECK-SAME: flagtree.debug.scope_id = 1 : i32
// CHECK: arith.addf
// CHECK-SAME: flagtree.debug.addr_level = 0 : i32
// CHECK-SAME: flagtree.debug.is_memory_op = false
// CHECK-SAME: flagtree.debug.op_id = 2 : i32
// CHECK-SAME: flagtree.debug.scope_id = 1 : i32
// CHECK: memref.store
// CHECK-SAME: flagtree.debug.addr_level = 0 : i32
// CHECK-SAME: flagtree.debug.is_memory_op = true
// CHECK-SAME: flagtree.debug.op_category = "store"
// CHECK-SAME: flagtree.debug.op_id = 3 : i32
// CHECK: arith.addf
// CHECK-NOT: flagtree.debug.op_id

// -----

module {
  func.func @two_scopes(%arg0: memref<4xf32>, %idx: index) {
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
    %0 = memref.load %arg0[%idx] : memref<4xf32>
    "flagtree_debug.collect_end"() : () -> ()
    "flagtree_debug.collect_begin"() {level = 2 : i32} : () -> ()
    %1 = arith.addf %0, %0 : f32
    "flagtree_debug.collect_end"() : () -> ()
    func.return
  }
}

// CHECK-LABEL: func.func @two_scopes
// CHECK: memref.load
// CHECK-SAME: flagtree.debug.addr_level = 0 : i32
// CHECK-SAME: flagtree.debug.op_id = 1 : i32
// CHECK-SAME: flagtree.debug.record_level = 1 : i32
// CHECK-SAME: flagtree.debug.scope_id = 1 : i32
// CHECK: arith.addf
// CHECK-SAME: flagtree.debug.addr_level = 0 : i32
// CHECK-SAME: flagtree.debug.op_id = 2 : i32
// CHECK-SAME: flagtree.debug.record_level = 2 : i32
// CHECK-SAME: flagtree.debug.scope_id = 2 : i32

// -----

module attributes {"flagtree.debug.addr_level" = 1 : i32} {
  func.func @addr_level_scope_override(%arg0: memref<4xf32>, %idx: index) {
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
    %0 = memref.load %arg0[%idx] : memref<4xf32>
    "flagtree_debug.collect_end"() : () -> ()
    "flagtree_debug.collect_begin"() {addr_level = 0 : i32, level = 1 : i32} : () -> ()
    %1 = arith.addf %0, %0 : f32
    "flagtree_debug.collect_end"() : () -> ()
    func.return
  }
}

// CHECK-LABEL: func.func @addr_level_scope_override
// CHECK: memref.load
// CHECK-SAME: flagtree.debug.addr_level = 1 : i32
// CHECK-SAME: flagtree.debug.op_id = 1 : i32
// CHECK-SAME: flagtree.debug.scope_id = 1 : i32
// CHECK: arith.addf
// CHECK-SAME: flagtree.debug.addr_level = 0 : i32
// CHECK-SAME: flagtree.debug.op_id = 2 : i32
// CHECK-SAME: flagtree.debug.scope_id = 2 : i32
