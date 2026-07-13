// RUN: triton-opt %s -split-input-file --flagtree-resolve-debug-scope --flagtree-assign-debug-op-id | FileCheck %s

module attributes {flagtree.debug.kernel_id_seed = "89abcdef01234567"} {
  func.func @same_kernel(%arg0: memref<4xf32>, %idx: index) {
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
    %0 = memref.load %arg0[%idx] : memref<4xf32>
    "flagtree_debug.collect_end"() : () -> ()
    func.return
  }
}

// CHECK-LABEL: module attributes
// CHECK-SAME: flagtree.debug.kernel_id = 2309737967 : i64
// CHECK-SAME: \22debugKernelId\22:2309737967
// CHECK-LABEL: func.func @same_kernel

// -----

module attributes {flagtree.debug.kernel_id_seed = "0000000200000000"} {
  func.func @same_kernel(%arg0: memref<4xf32>, %idx: index) {
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
    %0 = memref.load %arg0[%idx] : memref<4xf32>
    "flagtree_debug.collect_end"() : () -> ()
    func.return
  }
}

// CHECK-LABEL: module attributes
// CHECK-SAME: flagtree.debug.kernel_id = 2 : i64
// CHECK-SAME: \22debugKernelId\22:2
// CHECK-LABEL: func.func @same_kernel

// -----

module {
  tt.func @tt_memory_metadata(%ptr: !tt.ptr<f32>, %mask: i1) {
    "flagtree_debug.collect_begin"() {level = 1 : i32} : () -> ()
    %0 = tt.load %ptr : !tt.ptr<f32>
    tt.store %ptr, %0, %mask : !tt.ptr<f32>
    "flagtree_debug.collect_end"() : () -> ()
    tt.return
  }
}

// CHECK-LABEL: module attributes
// CHECK-SAME: \22kernelName\22:\22tt_memory_metadata\22
// CHECK-SAME: \22scopeCount\22:1
// CHECK-SAME: \22trackedOpCount\22:2
// CHECK-SAME: {\22accessBytes\22:4,\22accessType\22:\22load\22,\22addrSpace\22:\22global\22
// CHECK-SAME: \22mlirOpName\22:\22tt.load\22
// CHECK-SAME: \22operandRole\22:\22ptr\22,\22producerOpId\22:0,\22value\22:{\22addrSpace\22:\22global\22,\22dtype\22:\22!tt.ptr<f32>\22
// CHECK-SAME: \22result\22:{\22addrSpace\22:\22\22,\22dtype\22:\22f32\22,\22elementBits\22:32,\22elementDtype\22:\22f32\22
// CHECK-SAME: {\22accessBytes\22:4,\22accessType\22:\22store\22,\22addrSpace\22:\22global\22
// CHECK-SAME: \22mlirOpName\22:\22tt.store\22
// CHECK-SAME: \22operandRole\22:\22value\22,\22producerOpId\22:1
// CHECK-SAME: \22isPredicate\22:true,\22operandIndex\22:2
// CHECK-SAME: flagtree.debug.scope_count = 1 : i32
// CHECK-SAME: flagtree.debug.tracked_op_count = 2 : i32
// CHECK-LABEL: tt.func @tt_memory_metadata
// CHECK: tt.load
// CHECK-SAME: flagtree.debug.op_id = 1 : i32
// CHECK: tt.store
// CHECK-SAME: flagtree.debug.op_id = 2 : i32
