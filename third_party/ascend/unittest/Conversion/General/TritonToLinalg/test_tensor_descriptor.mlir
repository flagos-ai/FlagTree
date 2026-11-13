// RUN: triton-opt --triton-rewrite-tensor-descriptor-to-pointer --split-input-file %s | FileCheck %s --check-prefix=CHECK
// RUN: triton-opt --triton-rewrite-tensor-descriptor-to-pointer --canonicalize "--triton-to-linalg=global-kernel=false named-ops=True" --split-input-file %s | FileCheck %s --check-prefix=CHECK-LINALG

// -----

// Case 1: tt.descriptor_load/store -> tt.load/store (rewrite_tensor_descriptor_to_pointer)
// CHECK-LABEL: tt.func public @desc_copy_device
// CHECK-NOT: tt.make_tensor_descriptor
// CHECK-NOT: tt.descriptor_load
// CHECK-NOT: tt.descriptor_store
// CHECK-NOT: !tt.tensordesc
// CHECK: tt.load {{.*}}, {{.*}}, {{.*}} : tensor<16x32x!tt.ptr<f32>>
// CHECK: tt.store
//
// CHECK-LINALG-LABEL: func.func @desc_copy_device
// CHECK-LINALG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-LINALG: %[[ALLOC:.*]] = memref.alloc() : memref<16x32xf32>
// CHECK-LINALG: scf.if
// CHECK-LINALG: linalg.fill ins(%[[ZERO]] : f32) outs(%[[ALLOC]] : memref<16x32xf32>)
// CHECK-LINALG: memref.copy
// CHECK-LINALG: bufferization.to_tensor %[[ALLOC]]

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @desc_copy_device(%in_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>, %M: i32, %N: i32) {
    %c1 = arith.constant 1 : i64
    %c0 = arith.constant 0 : i32
    %Ns = arith.extsi %N : i32 to i64
    %in_desc = tt.make_tensor_descriptor %in_ptr, [%M, %N], [%Ns, %c1] : <f32>, <tensor<16x32xf32>>
    %out_desc = tt.make_tensor_descriptor %out_ptr, [%M, %N], [%Ns, %c1] : <f32>, <tensor<16x32xf32>>
    %tile = tt.descriptor_load %in_desc[%c0, %c0] : !tt.tensordesc<tensor<16x32xf32>> -> tensor<16x32xf32>
    tt.descriptor_store %out_desc[%c0, %c0], %tile : !tt.tensordesc<tensor<16x32xf32>>, tensor<16x32xf32>
    tt.return
  }
}

// -----

// Case 2: host TensorDescriptor ABI (padding as i1) -> linalg, getScalarValue peels
//   other = select(padding, nan, zero) into a scalar fill.
// CHECK-LABEL: tt.func public @host_desc_load
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : tensor<16x32xf32>
// CHECK: tt.load {{.*}}, {{.*}}, {{.*}} : tensor<16x32x!tt.ptr<f32>>
// CHECK: tt.store
//
// CHECK-LINALG-LABEL: func.func @host_desc_load
// CHECK-LINALG-DAG: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-LINALG-DAG: %[[NAN:.*]] = arith.constant 0x7FC00000 : f32
// CHECK-LINALG-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<16x32xf32>
// CHECK-LINALG: %[[PAD:.*]] = arith.select %{{.*}}, %[[NAN]], %[[ZERO]] : f32
// CHECK-LINALG: scf.if
// CHECK-LINALG: linalg.fill ins(%[[PAD]] : f32) outs(%[[ALLOC]] : memref<16x32xf32>)
// CHECK-LINALG: memref.copy
// CHECK-LINALG: bufferization.to_tensor %[[ALLOC]]

module attributes {hacc.target = #hacc.target<"Ascend910B4">} {
  tt.func public @host_desc_load(
      %ptr: !tt.ptr<f32>,
      %shape0: i64, %shape1: i64,
      %stride0: i64, %stride1: i64,
      %padding: i1,
      %shape0_i32: i32, %shape1_i32: i32,
      %stride0_dup: i64, %stride1_dup: i64,
      %out: !tt.ptr<f32>,
      %om: i32, %on: i32) {
    %cst_nan = arith.constant dense<0x7FC00000> : tensor<16x32xf32>
    %cst_zero = arith.constant dense<0.000000e+00> : tensor<16x32xf32>
    %other = arith.select %padding, %cst_nan, %cst_zero : tensor<16x32xf32>

    %om64 = arith.extsi %om : i32 to i64
    %on64 = arith.extsi %on : i32 to i64
    %offs_m = tt.splat %om64 : i64 -> tensor<16xi64>
    %range_m = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %range_m64 = arith.extsi %range_m : tensor<16xi32> to tensor<16xi64>
    %idx_m = arith.addi %offs_m, %range_m64 : tensor<16xi64>
    %idx_m2 = tt.expand_dims %idx_m {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>

    %offs_n = tt.splat %on64 : i64 -> tensor<32xi64>
    %range_n = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %range_n64 = arith.extsi %range_n : tensor<32xi32> to tensor<32xi64>
    %idx_n = arith.addi %offs_n, %range_n64 : tensor<32xi64>
    %idx_n2 = tt.expand_dims %idx_n {axis = 0 : i32} : tensor<32xi64> -> tensor<1x32xi64>

    %base = tt.splat %ptr : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
    %s0 = tt.splat %stride0 : i64 -> tensor<16x1xi64>
    %m0 = arith.muli %idx_m2, %s0 : tensor<16x1xi64>
    %b0 = tt.broadcast %m0 : tensor<16x1xi64> -> tensor<16x32xi64>
    %s1 = tt.splat %stride1 : i64 -> tensor<1x32xi64>
    %m1 = arith.muli %idx_n2, %s1 : tensor<1x32xi64>
    %b1 = tt.broadcast %m1 : tensor<1x32xi64> -> tensor<16x32xi64>
    %off = arith.addi %b0, %b1 : tensor<16x32xi64>
    %ptrs = tt.addptr %base, %off : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi64>

    %c0 = arith.constant dense<0> : tensor<16x1xi64>
    %ge0 = arith.cmpi sge, %idx_m2, %c0 : tensor<16x1xi64>
    %sh0 = tt.splat %shape0 : i64 -> tensor<16x1xi64>
    %lt0 = arith.cmpi slt, %idx_m2, %sh0 : tensor<16x1xi64>
    %mmsk = arith.andi %ge0, %lt0 : tensor<16x1xi1>
    %mmsk2 = tt.broadcast %mmsk : tensor<16x1xi1> -> tensor<16x32xi1>
    %c0n = arith.constant dense<0> : tensor<1x32xi64>
    %ge1 = arith.cmpi sge, %idx_n2, %c0n : tensor<1x32xi64>
    %sh1 = tt.splat %shape1 : i64 -> tensor<1x32xi64>
    %lt1 = arith.cmpi slt, %idx_n2, %sh1 : tensor<1x32xi64>
    %nmsk = arith.andi %ge1, %lt1 : tensor<1x32xi1>
    %nmsk2 = tt.broadcast %nmsk : tensor<1x32xi1> -> tensor<16x32xi1>
    %mask = arith.andi %mmsk2, %nmsk2 : tensor<16x32xi1>

    %tile = tt.load %ptrs, %mask, %other : tensor<16x32x!tt.ptr<f32>>

    %obase = tt.splat %out : !tt.ptr<f32> -> tensor<16x32x!tt.ptr<f32>>
    %optrs = tt.addptr %obase, %off : tensor<16x32x!tt.ptr<f32>>, tensor<16x32xi64>
    tt.store %optrs, %tile, %mask : tensor<16x32x!tt.ptr<f32>>
    tt.return
  }
}
