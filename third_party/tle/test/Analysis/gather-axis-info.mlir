// Copyright 2026 FlagOS Contributors
// RUN: triton-opt %s -test-print-alignment 2>&1 | FileCheck %s

// Arbitrary gather indices preserve a divisor that holds for every source
// element. This is the row-offset pattern used by the masked BF16 PV load.
tt.func @gather_strided_rows(%indices: tensor<64xi32>) {
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  %stride = arith.constant dense<512> : tensor<64xi32>
  %rows = arith.muli %range, %stride : tensor<64xi32>
  // CHECK: tt.gather {{.*}} => contiguity = [1], divisibility = [512], constancy = [1], constant_value = <none>
  %gather = tt.gather %rows[%indices] {axis = 0 : i32} : (tensor<64xi32>, tensor<64xi32>) -> tensor<64xi32>
  tt.return
}

// A contiguous group may have an aligned first element and unaligned members.
// Neither that group-start divisor nor contiguity survives arbitrary indices.
tt.func @gather_contiguous_range(%indices: tensor<32xi32>) {
  %range = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32>
  // CHECK: tt.gather {{.*}} => contiguity = [1], divisibility = [1], constancy = [1], constant_value = <none>
  %gather = tt.gather %range[%indices] {axis = 0 : i32} : (tensor<64xi32>, tensor<32xi32>) -> tensor<32xi32>
  tt.return
}

// The strongest universal power-of-two divisor can be proved on any axis.
tt.func @gather_rank_two(%src: tensor<8x16xi64> {tt.divisibility = dense<[8, 16]> : tensor<2xi32>}, %indices: tensor<8x8xi32>) {
  // CHECK: tt.gather {{.*}} => contiguity = [1, 1], divisibility = [16, 16], constancy = [1, 1], constant_value = <none>
  %gather = tt.gather %src[%indices] {axis = 1 : i32} : (tensor<8x16xi64>, tensor<8x8xi32>) -> tensor<8x8xi64>
  tt.return
}

tt.func @gather_rank_two_contiguous(%src: tensor<8x16xi32> {tt.contiguity = dense<[1, 4]> : tensor<2xi32>, tt.divisibility = dense<[1, 16]> : tensor<2xi32>}, %indices: tensor<4x16xi32>) {
  // CHECK: tt.gather {{.*}} => contiguity = [1, 1], divisibility = [1, 1], constancy = [1, 1], constant_value = <none>
  %gather = tt.gather %src[%indices] {axis = 0 : i32} : (tensor<8x16xi32>, tensor<4x16xi32>) -> tensor<4x16xi32>
  tt.return
}

tt.func @gather_uniform_constant(%indices: tensor<8x8xi32>) {
  %src = arith.constant dense<32> : tensor<8x16xi32>
  // CHECK: tt.gather {{.*}} => contiguity = [1, 1], divisibility = [32, 32], constancy = [8, 8], constant_value = 32
  %gather = tt.gather %src[%indices] {axis = 1 : i32} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<8x8xi32>
  tt.return
}

tt.func @gather_uniform_scalar(%scalar: i32 {tt.divisibility = 16 : i32}, %indices: tensor<8x8xi32>) {
  %src = tt.splat %scalar : i32 -> tensor<8x16xi32>
  // CHECK: tt.gather {{.*}} => contiguity = [1, 1], divisibility = [16, 16], constancy = [8, 8], constant_value = <none>
  %gather = tt.gather %src[%indices] {axis = 1 : i32} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<8x8xi32>
  tt.return
}

// Uniform source rows are insufficient when different rows can select
// different column indices. Do not invent output constancy from one axis.
tt.func @gather_partial_constancy(%src: tensor<8x16xi32> {tt.constancy = dense<[8, 1]> : tensor<2xi32>, tt.divisibility = dense<[2, 2]> : tensor<2xi32>}, %indices: tensor<8x8xi32>) {
  // CHECK: tt.gather {{.*}} => contiguity = [1, 1], divisibility = [2, 2], constancy = [1, 1], constant_value = <none>
  %gather = tt.gather %src[%indices] {axis = 1 : i32} : (tensor<8x16xi32>, tensor<8x8xi32>) -> tensor<8x8xi32>
  tt.return
}

tt.func @gather_rank_three(%src: tensor<4x16x8xi32> {tt.divisibility = dense<[4, 16, 8]> : tensor<3xi32>}, %indices: tensor<4x8x8xi32>) {
  // CHECK: tt.gather {{.*}} => contiguity = [1, 1, 1], divisibility = [16, 16, 16], constancy = [1, 1, 1], constant_value = <none>
  %gather = tt.gather %src[%indices] {axis = 1 : i32} : (tensor<4x16x8xi32>, tensor<4x8x8xi32>) -> tensor<4x8x8xi32>
  tt.return
}

tt.func @gather_aligned_pointers(%src: tensor<64x!tt.ptr<bf16>> {tt.divisibility = 16 : i32}, %indices: tensor<32xi32>) {
  // CHECK: tt.gather {{.*}} => contiguity = [1], divisibility = [16], constancy = [1], constant_value = <none>
  %gather = tt.gather %src[%indices] {axis = 0 : i32} : (tensor<64x!tt.ptr<bf16>>, tensor<32xi32>) -> tensor<32x!tt.ptr<bf16>>
  tt.return
}
