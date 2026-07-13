// RUN: triton-opt %s --flagtree-insert-debug-records | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {flagtree.debug.addr_level = 1 : i32, flagtree.debug.record_level = 1 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: tt.func @descriptor_addresses
  // CHECK-SAME: (%[[GLOBAL:[a-zA-Z0-9_]+]]: !tt.ptr<i16>, %[[DIM:[a-zA-Z0-9_]+]]: i32, %[[STRIDE:[a-zA-Z0-9_]+]]: i64)
  tt.func @descriptor_addresses(%global: !tt.ptr<i16>, %dim: i32, %stride: i64) attributes {flagtree.debug.hidden_arg = "__debug_ctrl_ptr", flagtree.debug.logical_instance_id_formula = "pid0 + pid1 * num_programs0 + pid2 * num_programs0 * num_programs1"} {
    // CHECK: %[[DESC:.*]] = tt.make_tensor_descriptor %[[GLOBAL]], [%[[DIM]]], [%[[STRIDE]]]
    %desc = tt.make_tensor_descriptor %global, [%dim], [%stride] : !tt.ptr<i16>, !tt.tensordesc<tensor<32xi16>>
    %c0_i32 = arith.constant 0 : i32
    // CHECK: tt.descriptor_load %[[DESC]]
    // CHECK-SAME: flagtree.debug.instrumented = true
    %value = tt.descriptor_load %desc[%c0_i32] {flagtree.debug.is_memory_op = true, flagtree.debug.op_id = 31 : i32} : !tt.tensordesc<tensor<32xi16>> -> tensor<32xi16, #blocked>
    // CHECK: "flagtree_debug.capture_memory_address"(%[[GLOBAL]])
    // CHECK-SAME: op_id = 31 : i32
    // CHECK-SAME: operand_index = 0 : i32
    tt.return
  }
}
