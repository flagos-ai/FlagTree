// RUN: triton-shared-opt --split-input-file %s 2>&1 | FileCheck %s
//
// Verifies xsmt::MemorySpaceAttr accepts the four canonical names that
// spine-mlir's XSMT dialect expects (see spine-mlir
// lib/Common/Dialect/XSMT/IR/XSMT.cc): "global", "thread_local",
// "cluster_shared_l2", "fragment". Any other value must be rejected by
// the attribute's verify hook.
//
// The IR syntax is `#xsmt.memory_space<"name">` (StringRef parameter
// rendered as a quoted string).
//
// Output ordering: when --split-input-file is used with `2>&1`, the
// stderr from invalid chunks (unbuffered) lands before the stdout from
// valid chunks (buffered). The CHECKs below follow that order.

// invalid: legacy frontend name "tcm" must be rejected
// CHECK: unsupported xsmt memory space "tcm"
func.func @invalid_tcm(%arg0 : memref<4x4xf32, #xsmt.memory_space<"tcm">>) {
  return
}

// -----

// invalid: legacy frontend name "l2" must be rejected
// CHECK: unsupported xsmt memory space "l2"
func.func @invalid_l2(%arg0 : memref<4x4xf32, #xsmt.memory_space<"l2">>) {
  return
}

// -----

// invalid: bogus name must be rejected
// CHECK: unsupported xsmt memory space "bogus"
func.func @invalid_bogus(%arg0 : memref<4x4xf32, #xsmt.memory_space<"bogus">>) {
  return
}

// -----

// valid: global
// CHECK-LABEL: func @valid_global
// CHECK: memref<4x4xf32, #xsmt.memory_space<"global">>
func.func @valid_global(%arg0 : memref<4x4xf32, #xsmt.memory_space<"global">>) {
  return
}

// -----

// valid: thread_local
// CHECK-LABEL: func @valid_thread_local
// CHECK: memref<4x4xf32, #xsmt.memory_space<"thread_local">>
func.func @valid_thread_local(%arg0 : memref<4x4xf32, #xsmt.memory_space<"thread_local">>) {
  return
}

// -----

// valid: cluster_shared_l2
// CHECK-LABEL: func @valid_cluster_shared_l2
// CHECK: memref<4x4xf32, #xsmt.memory_space<"cluster_shared_l2">>
func.func @valid_cluster_shared_l2(%arg0 : memref<4x4xf32, #xsmt.memory_space<"cluster_shared_l2">>) {
  return
}

// -----

// valid: fragment
// CHECK-LABEL: func @valid_fragment
// CHECK: memref<4x4xf32, #xsmt.memory_space<"fragment">>
func.func @valid_fragment(%arg0 : memref<4x4xf32, #xsmt.memory_space<"fragment">>) {
  return
}
