// RUN: triton-opt %s -split-input-file --flagtree-insert-debug-records | FileCheck %s

module attributes {flagtree.debug.addr_level = 1 : i32, flagtree.debug.record_level = 1 : i32} {
  // CHECK-LABEL: func.func @summary_and_memory
  // CHECK-SAME: flagtree.debug.hidden_arg = "__debug_ctrl_ptr"
  // CHECK-SAME: flagtree.debug.logical_instance_id_formula = "pid0 + pid1 * num_programs0 + pid2 * num_programs0 * num_programs1"
  func.func @summary_and_memory(%arg0: memref<1xf32>, %arg1: memref<1xf32>, %idx: index) {
    // CHECK: %[[LOAD:.*]] = memref.load %arg0[%arg{{[0-9]+}}]
    // CHECK-SAME: flagtree.debug.instrumented = true
    // CHECK-SAME: flagtree.debug.memory_event_kind = "LAST_ALIGNED_ADDR"
    // CHECK-SAME: flagtree.debug.record_kinds = ["summary", "memory_event"]
    // CHECK-SAME: flagtree.debug.summary_collectors = ["nan_count", "inf_count", "zero_count", "mean_finite", "min_finite", "max_finite", "l2_norm", "element_count"]
    %0 = memref.load %arg0[%idx] {flagtree.debug.op_id = 7 : i32, flagtree.debug.is_memory_op = true} : memref<1xf32>
    // CHECK: "flagtree_debug.record_summary_bundle"(%[[LOAD]])
    // CHECK-SAME: collectors = ["nan_count", "inf_count", "zero_count", "mean_finite", "min_finite", "max_finite", "l2_norm", "element_count"]
    // CHECK-SAME: op_id = 7 : i32
    // CHECK-SAME: record_level = 1 : i32
    // CHECK: "flagtree_debug.capture_memory_address"(%arg0)
    // CHECK-SAME: event_kind = "LAST_ALIGNED_ADDR"
    // CHECK-SAME: lowering_policy = "backend_sensitive"
    // CHECK-SAME: op_id = 7 : i32
    // CHECK-SAME: operand_index = 0 : i32

    // CHECK: %[[ADD:.*]] = arith.addf %[[LOAD]], %[[LOAD]]
    // CHECK-SAME: flagtree.debug.instrumented = true
    // CHECK-SAME: flagtree.debug.record_kinds = ["summary"]
    // CHECK-SAME: flagtree.debug.summary_collectors = ["nan_count", "inf_count", "zero_count", "mean_finite", "min_finite", "max_finite", "l2_norm", "element_count"]
    %1 = arith.addf %0, %0 {flagtree.debug.op_id = 8 : i32} : f32
    // CHECK: "flagtree_debug.record_summary_bundle"(%[[ADD]])
    // CHECK-SAME: op_id = 8 : i32

    // CHECK: memref.store %[[ADD]], %arg1[%arg{{[0-9]+}}]
    // CHECK-SAME: flagtree.debug.instrumented = true
    // CHECK-SAME: flagtree.debug.memory_event_kind = "LAST_ALIGNED_ADDR"
    // CHECK-SAME: flagtree.debug.record_kinds = ["memory_event"]
    memref.store %1, %arg1[%idx] {flagtree.debug.op_id = 9 : i32, flagtree.debug.is_memory_op = true} : memref<1xf32>
    // CHECK: "flagtree_debug.capture_memory_address"(%arg1)
    // CHECK-SAME: event_kind = "LAST_ALIGNED_ADDR"
    // CHECK-SAME: op_id = 9 : i32
    // CHECK-SAME: operand_index = 1 : i32
    func.return
  }
}

// -----

module attributes {flagtree.debug.addr_level = 0 : i32, flagtree.debug.record_level = 1 : i32} {
  // CHECK-LABEL: func.func @addr_level_zero_summary_only
  // CHECK-SAME: flagtree.debug.hidden_arg = "__debug_ctrl_ptr"
  func.func @addr_level_zero_summary_only(%arg0: memref<1xf32>, %idx: index) {
    // CHECK: %[[LOAD:.*]] = memref.load %arg0[%arg{{[0-9]+}}]
    // CHECK-SAME: flagtree.debug.instrumented = true
    // CHECK-SAME: flagtree.debug.record_kinds = ["summary"]
    // CHECK-NOT: flagtree.debug.memory_event_kind
    %0 = memref.load %arg0[%idx] {flagtree.debug.op_id = 10 : i32, flagtree.debug.is_memory_op = true} : memref<1xf32>
    // CHECK: "flagtree_debug.record_summary_bundle"(%[[LOAD]])
    // CHECK-SAME: op_id = 10 : i32
    // CHECK-NOT: "flagtree_debug.capture_memory_address"
    func.return
  }
}

// -----

module attributes {flagtree.debug.addr_level = 1 : i32, flagtree.debug.record_level = 2 : i32} {
  // CHECK-LABEL: func.func @tensor_full
  // CHECK-SAME: flagtree.debug.hidden_arg = "__debug_ctrl_ptr"
  func.func @tensor_full(%arg0: memref<1xf32>, %idx: index) {
    // CHECK: %[[LOAD:.*]] = memref.load %arg0[%arg{{[0-9]+}}]
    // CHECK-SAME: flagtree.debug.full_value_ref = true
    // CHECK-SAME: flagtree.debug.record_kinds = ["summary", "memory_event", "full_value"]
    %0 = memref.load %arg0[%idx] {flagtree.debug.op_id = 11 : i32, flagtree.debug.is_memory_op = true} : memref<1xf32>
    // CHECK: "flagtree_debug.record_summary_bundle"(%[[LOAD]])
    // CHECK-SAME: record_level = 2 : i32
    // CHECK: "flagtree_debug.capture_memory_address"(%arg0)
    // CHECK-SAME: op_id = 11 : i32
    // CHECK: "flagtree_debug.record_full_value_ref"(%[[LOAD]])
    // CHECK-SAME: op_id = 11 : i32
    // CHECK-SAME: record_level = 2 : i32
    func.return
  }
}

// -----

module attributes {flagtree.debug.addr_level = 1 : i32, flagtree.debug.enable_hidden_arg_abi = true, flagtree.debug.record_level = 1 : i32} {
  // CHECK: flagtree.debug.records_per_instance = 8 : i32
  // CHECK-LABEL: tt.func @tt_hidden_arg_abi
  // CHECK-SAME: %arg0: !tt.ptr<f32>
  // CHECK-SAME: %arg1: !tt.ptr<i32>
  // CHECK-SAME: flagtree.debug.hidden_arg = "__debug_ctrl_ptr"
  // CHECK-SAME: flagtree.debug.hidden_arg_index = 1 : i32
  // CHECK-SAME: flagtree.debug.hidden_arg_type = "!tt.ptr<i32>"
  tt.func @tt_hidden_arg_abi(%ptr: !tt.ptr<f32>) {
    // CHECK: tt.get_program_id x
    // CHECK: tt.get_num_programs x
    // CHECK: %[[LOAD:.*]] = tt.load %arg0
    %0 = tt.load %ptr {flagtree.debug.op_id = 21 : i32, flagtree.debug.scope_id = 1 : i32, flagtree.debug.is_memory_op = true} : !tt.ptr<f32>
    // CHECK: tt.ptr_to_int %arg0
    // CHECK: arith.constant 3 : i64
    // CHECK: arith.constant 8 : i64
    // CHECK: tt.store
    // CHECK-NOT: tt.atomic_rmw
    // CHECK-NOT: tensor<8x!tt.ptr<i32>>
    // CHECK-NOT: "flagtree_debug.record_summary_bundle"
    // CHECK-NOT: "flagtree_debug.record_summary"
    // CHECK-NOT: "flagtree_debug.record_memory_event"
    // CHECK-NOT: "flagtree_debug.capture_memory_address"
    tt.return
  }
}

// -----

module attributes {flagtree.debug.addr_level = 0 : i32, flagtree.debug.enable_hidden_arg_abi = true, flagtree.debug.record_level = 1 : i32, flagtree.debug.timeline_enabled = true, flagtree.debug.timeline_only = true} {
  // CHECK-LABEL: tt.func @tt_timeline_uses_contiguous_ring_access
  tt.func @tt_timeline_uses_contiguous_ring_access(%ptr: !tt.ptr<f32>) {
    // Header fields are extracted from one aligned 32-byte load.
    // CHECK: tt.load {{.*}} : tensor<8x!tt.ptr<i32>>
    // CHECK: tensor.extract {{.*}} : tensor<8xi32>
    // CHECK: tt.load {{.*}} : tensor<8x!tt.ptr<i32>>
    // CHECK: tensor.extract {{.*}} : tensor<8xi32>
    // CHECK: %[[LOAD:.*]] = tt.load %arg0
    %0 = tt.load %ptr {flagtree.debug.op_id = 51 : i32, flagtree.debug.scope_id = 1 : i32, flagtree.debug.is_memory_op = true} : !tt.ptr<f32>
    // A timeline record is committed as one guarded 64-byte store, rather
    // than scalar GM stores that are illegal on the Ascend AIV path.
    // CHECK: scf.if
    // CHECK: tt.store {{.*}} : tensor<16x!tt.ptr<i32>>
    // CHECK-NOT: tt.store {{.*}} : !tt.ptr<i32>
    tt.return
  }
}

// -----

module attributes {flagtree.debug.addr_level = 1 : i32, flagtree.debug.record_level = 1 : i32} {
  // CHECK-LABEL: tt.func @tt_address_summary
  tt.func @tt_address_summary(%ptr: !tt.ptr<f32>, %n: i32) {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %limit = tt.splat %n : i32 -> tensor<16xi32>
    // CHECK: %[[MASK:.*]] = arith.cmpi slt
    %mask = arith.cmpi slt, %range, %limit : tensor<16xi32>
    // CHECK: %[[PTRS:.*]] = tt.addptr
    %ptrs = tt.addptr %base, %range : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
    // CHECK: %[[LOAD:.*]] = tt.load %[[PTRS]], %[[MASK]]
    // CHECK-SAME: flagtree.debug.memory_event_kind = "ADDRESS_SUMMARY"
    %0 = tt.load %ptrs, %mask, %cst {flagtree.debug.op_id = 41 : i32, flagtree.debug.scope_id = 1 : i32, flagtree.debug.is_memory_op = true} : tensor<16x!tt.ptr<f32>>
    // CHECK: "flagtree_debug.capture_memory_address"(%[[PTRS]], %[[MASK]])
    // CHECK-SAME: event_kind = "FIRST_ADDR"
    // CHECK-SAME: lowering_policy = "cann9_address_summary"
    // CHECK-SAME: access_bytes = 4 : i32
    // CHECK: "flagtree_debug.capture_memory_address"(%[[PTRS]], %[[MASK]])
    // CHECK-SAME: event_kind = "LAST_ADDR"
    // CHECK: "flagtree_debug.capture_memory_address"(%[[PTRS]], %[[MASK]])
    // CHECK-SAME: event_kind = "MIN_ADDR"
    // CHECK: "flagtree_debug.capture_memory_address"(%[[PTRS]], %[[MASK]])
    // CHECK-SAME: event_kind = "MAX_ADDR"
    // CHECK: "flagtree_debug.capture_memory_address"(%[[PTRS]], %[[MASK]])
    // CHECK-SAME: event_kind = "ACTIVE_LANE_COUNT"
    // CHECK: "flagtree_debug.capture_memory_address"(%[[PTRS]], %[[MASK]])
    // CHECK-SAME: event_kind = "ADDRESS_SPAN_BYTES"
    tt.return
  }
}

// -----

module attributes {flagtree.debug.addr_level = 1 : i32, flagtree.debug.record_level = 1 : i32} {
  // CHECK-LABEL: tt.func @tt_large_load_level1_is_memory_only
  tt.func @tt_large_load_level1_is_memory_only(%ptr: !tt.ptr<f32>) {
    // CHECK: %[[BASE:.*]] = tt.splat
    %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    // CHECK: %[[LOAD:.*]] = tt.load %[[BASE]]
    // CHECK-SAME: flagtree.debug.memory_event_kind = "BASE_ALIGNED_ADDR"
    // CHECK-SAME: flagtree.debug.record_kinds = ["memory_event"]
    // CHECK-NOT: flagtree.debug.summary_collectors
    %0 = tt.load %base {flagtree.debug.op_id = 43 : i32, flagtree.debug.scope_id = 1 : i32, flagtree.debug.is_memory_op = true} : tensor<512x!tt.ptr<f32>>
    // CHECK-NOT: "flagtree_debug.record_summary_bundle"
    // CHECK-NOT: "flagtree_debug.record_summary"
    // CHECK: "flagtree_debug.capture_memory_address"(%[[BASE]])
    // CHECK-SAME: event_kind = "BASE_ALIGNED_ADDR"
    tt.return
  }
}

// -----

module attributes {flagtree.debug.addr_level = 1 : i32, flagtree.debug.enable_hidden_arg_abi = true, flagtree.debug.record_level = 1 : i32} {
  // CHECK-LABEL: tt.func @tt_hidden_arg_abi_tensor_ptr
  tt.func @tt_hidden_arg_abi_tensor_ptr(%ptrs: tensor<16x!tt.ptr<f32>>) {
    // CHECK: %[[LOAD:.*]] = tt.load %arg0
    // CHECK-SAME: flagtree.debug.memory_event_kind = "BASE_ALIGNED_ADDR"
    %0 = tt.load %ptrs {flagtree.debug.op_id = 31 : i32, flagtree.debug.scope_id = 1 : i32, flagtree.debug.is_memory_op = true} : tensor<16x!tt.ptr<f32>>
    // CHECK-NOT: linalg.generic
    // CHECK-NOT: tt.ptr_to_int %arg0 : tensor<16x!tt.ptr<f32>> -> tensor<16xi64>
    // CHECK-NOT: "flagtree_debug.capture_memory_address"
    tt.return
  }
}
