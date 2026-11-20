// RUN: triton-opt %s --triton-to-linalg -verify-each | FileCheck %s --implicit-check-not='tensor<' --implicit-check-not='arith.select'
// RUN: triton-opt %s --triton-to-linalg='named-ops=true' -verify-each | FileCheck %s --implicit-check-not='tensor<' --implicit-check-not='arith.select'
// RUN: sed 's/Ascend910B2/Ascend950DT_9582/' %s | triton-opt --triton-to-linalg='compile-on-910-95=true' -verify-each | FileCheck %s --implicit-check-not='tensor<' --implicit-check-not='arith.select'

// Scalar masked loads must keep their result scalar and only read memory in
// the active branch. In particular, selecting the result of an unconditional
// load does not protect an invalid masked-off address.
module attributes {hacc.target = #hacc.target<"Ascend910B2">} {
  // CHECK-LABEL: func.func @masked_i32(
  // CHECK-SAME: %[[MASK:[^:]+]]: i1, %[[OTHER:[^:]+]]: i32
  // CHECK-NOT: memref.load
  // CHECK: %[[RESULT:.*]] = scf.if %[[MASK]] -> (i32) {
  // CHECK-NEXT: %[[VALUE:.*]] = memref.load
  // CHECK-NEXT: scf.yield %[[VALUE]] : i32
  // CHECK-NEXT: } else {
  // CHECK-NEXT: scf.yield %[[OTHER]] : i32
  // CHECK-NEXT: }
  // CHECK-NOT: memref.load
  // CHECK: return %[[RESULT]] : i32
  tt.func public @masked_i32(%ptr: !tt.ptr<i32>, %mask: i1, %other: i32, %offset: i32) -> i32 {
    %address = tt.addptr %ptr, %offset : !tt.ptr<i32>, i32
    %value = tt.load %address, %mask, %other : !tt.ptr<i32>
    tt.return %value : i32
  }

  // CHECK-LABEL: func.func @masked_f32(
  // CHECK-SAME: %[[MASK:[^:]+]]: i1, %[[OTHER:[^:]+]]: f32
  // CHECK-NOT: memref.load
  // CHECK: %[[RESULT:.*]] = scf.if %[[MASK]] -> (f32) {
  // CHECK-NEXT: %[[VALUE:.*]] = memref.load
  // CHECK-NEXT: scf.yield %[[VALUE]] : f32
  // CHECK-NEXT: } else {
  // CHECK-NEXT: scf.yield %[[OTHER]] : f32
  // CHECK-NEXT: }
  // CHECK-NOT: memref.load
  // CHECK: return %[[RESULT]] : f32
  tt.func public @masked_f32(%ptr: !tt.ptr<f32>, %mask: i1, %other: f32) -> f32 {
    %value = tt.load %ptr, %mask, %other : !tt.ptr<f32>
    tt.return %value : f32
  }

  // No `other` leaves the inactive result unspecified, but the load must
  // still be conditional. Do not require a particular inactive result value.
  // CHECK-LABEL: func.func @masked_without_other(
  // CHECK-SAME: %[[MASK:[^:]+]]: i1
  // CHECK-NOT: memref.load
  // CHECK: %[[RESULT:.*]] = scf.if %[[MASK]] -> (i32) {
  // CHECK-NEXT: %[[VALUE:.*]] = memref.load
  // CHECK-NEXT: scf.yield %[[VALUE]] : i32
  // CHECK-NEXT: } else {
  // CHECK-NOT: memref.load
  // CHECK: scf.yield %{{.*}} : i32
  // CHECK-NEXT: }
  // CHECK-NOT: memref.load
  // CHECK: return %[[RESULT]] : i32
  tt.func public @masked_without_other(%ptr: !tt.ptr<i32>, %mask: i1) -> i32 {
    %value = tt.load %ptr, %mask : !tt.ptr<i32>
    tt.return %value : i32
  }

  // CHECK-LABEL: func.func @unmasked(
  // CHECK-NOT: scf.if
  // CHECK: %[[VALUE:.*]] = memref.load
  // CHECK-NOT: scf.if
  // CHECK: return %[[VALUE]] : i32
  tt.func public @unmasked(%ptr: !tt.ptr<i32>) -> i32 {
    %value = tt.load %ptr : !tt.ptr<i32>
    tt.return %value : i32
  }

  // CHECK-LABEL: func.func @mask_true(
  // CHECK-NOT: scf.if
  // CHECK: %[[VALUE:.*]] = memref.load
  // CHECK-NOT: scf.if
  // CHECK: return %[[VALUE]] : i32
  tt.func public @mask_true(%ptr: !tt.ptr<i32>, %other: i32) -> i32 {
    %true = arith.constant true
    %value = tt.load %ptr, %true, %other : !tt.ptr<i32>
    tt.return %value : i32
  }

  // CHECK-LABEL: func.func @mask_false(
  // CHECK-SAME: %[[OTHER:[^:]+]]: i32
  // CHECK-NOT: memref.load
  // CHECK-NOT: scf.if
  // CHECK: return %[[OTHER]] : i32
  tt.func public @mask_false(%ptr: !tt.ptr<i32>, %other: i32) -> i32 {
    %false = arith.constant false
    %value = tt.load %ptr, %false, %other : !tt.ptr<i32>
    tt.return %value : i32
  }

  // CHECK-LABEL: func.func @mask_false_without_other(
  // CHECK-NOT: memref.load
  // CHECK-NOT: scf.if
  // CHECK: return %{{.*}} : f32
  tt.func public @mask_false_without_other(%ptr: !tt.ptr<f32>) -> f32 {
    %false = arith.constant false
    %value = tt.load %ptr, %false : !tt.ptr<f32>
    tt.return %value : f32
  }
}
