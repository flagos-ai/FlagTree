// RUN: triton-opt '--discrete-mask-access-conversion=compile-on-910-95=True compile-mode=simd_simt_template' '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' --split-input-file %s | FileCheck %s

// CHECK-LABEL: tt.func public @structured_discrete_mask_atomic_add_1d
// CHECK: arith.select {{.*}} : tensor<8xi1>, tensor<8xi32>
// CHECK: tt.atomic_rmw add, acq_rel, gpu, {{.*}} : (tensor<8x!tt.ptr<i32>>, tensor<8xi32>) -> tensor<8xi32>
tt.func public @structured_discrete_mask_atomic_add_1d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
	%cst = arith.constant dense<2> : tensor<8xi32>
	%cst_0 = arith.constant dense<8> : tensor<8xi32>
	%0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
	%1 = arith.muli %0, %cst : tensor<8xi32>
	%2 = arith.cmpi slt, %1, %cst_0 : tensor<8xi32>
	%3 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
	%4 = tt.addptr %3, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
	%5 = tt.load %4 : tensor<8x!tt.ptr<i32>>
	%6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
	%7 = tt.addptr %6, %0 : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
	%8 = tt.atomic_rmw add, acq_rel, gpu, %7, %5, %2 : (tensor<8x!tt.ptr<i32>>, tensor<8xi32>, tensor<8xi1>) -> tensor<8xi32>
	tt.return
}

// -----

// CHECK-LABEL: tt.func public @structured_discrete_mask_atomic_add_i16_fallback
// CHECK: arith.select {{.*}} : tensor<8xi1>, tensor<8xi16>
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK: tt.atomic_rmw add, acq_rel, gpu, {{.*}} : (tensor<8x!tt.ptr<i16>>, tensor<8xi16>) -> tensor<8xi16>
tt.func public @structured_discrete_mask_atomic_add_i16_fallback(%arg0: !tt.ptr<i16>, %arg1: !tt.ptr<i16>) {
	%cst = arith.constant dense<2> : tensor<8xi32>
	%cst_0 = arith.constant dense<8> : tensor<8xi32>
	%0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
	%1 = arith.muli %0, %cst : tensor<8xi32>
	%2 = arith.cmpi slt, %1, %cst_0 : tensor<8xi32>
	%3 = tt.splat %arg0 : !tt.ptr<i16> -> tensor<8x!tt.ptr<i16>>
	%4 = tt.addptr %3, %0 : tensor<8x!tt.ptr<i16>>, tensor<8xi32>
	%5 = tt.load %4 : tensor<8x!tt.ptr<i16>>
	%6 = tt.splat %arg1 : !tt.ptr<i16> -> tensor<8x!tt.ptr<i16>>
	%7 = tt.addptr %6, %0 : tensor<8x!tt.ptr<i16>>, tensor<8xi32>
	%8 = tt.atomic_rmw add, acq_rel, gpu, %7, %5, %2 : (tensor<8x!tt.ptr<i16>>, tensor<8xi16>, tensor<8xi1>) -> tensor<8xi16>
	tt.return
}

// -----

// CHECK-LABEL: tt.func public @fully_unstructured_atomic_add_2d
// CHECK: %[[MASK_TRUE:.*]] = arith.constant dense<1> : tensor<16xi8>
// CHECK: %[[OFFSET_FLAT:.*]] = tt.reshape {{.*}} : tensor<4x4xi64> -> tensor<16xi64>
// CHECK: %[[VALUE_FLAT:.*]] = tt.reshape {{.*}} : tensor<4x4xi32> -> tensor<16xi32>
// CHECK: hivm.hir.custom {extra_attr = "operate=add"{{.*}}} "__builtin_indirect_atomic" ins(%arg2, %[[OFFSET_FLAT]], %[[VALUE_FLAT]], %[[MASK_TRUE]]
tt.func public @fully_unstructured_atomic_add_2d(%arg0: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
	%cst = arith.constant dense<true> : tensor<4x4xi1>
	%cst_0 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_0 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x4x!tt.ptr<i64>>
	%8 = tt.addptr %7, %6 : tensor<4x4x!tt.ptr<i64>>, tensor<4x4xi32>
	%9 = tt.load %8 : tensor<4x4x!tt.ptr<i64>>
	%10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%11 = tt.addptr %10, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%12 = tt.load %11 : tensor<4x4x!tt.ptr<i32>>
	%13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%14 = tt.addptr %13, %9 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi64>
	%15 = tt.atomic_rmw add, acq_rel, gpu, %14, %12, %cst : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%16 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%17 = tt.addptr %16, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %17, %15 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// sizeInByte >= 64, falls back to scf.for loop unrolling + DiscreteMemAccess
// CHECK-LABEL: tt.func public @partial_structured_atomic_add_2d
// CHECK: scf.for
// CHECK: tt.atomic_rmw add, acq_rel, gpu, {{.*}} {DiscreteMemAccess}
// CHECK: {ExtractedLoadOrStore}
tt.func public @partial_structured_atomic_add_2d(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
	%cst = arith.constant dense<true> : tensor<2x16xi1>
	%cst_0 = arith.constant dense<16> : tensor<2x1xi32>
	%cst_1 = arith.constant dense<2> : tensor<2x1xi32>
	%cst_2 = arith.constant dense<1> : tensor<2x1xi32>
	%0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
	%2 = arith.addi %1, %cst_2 : tensor<2x1xi32>
	%3 = arith.remsi %2, %cst_1 : tensor<2x1xi32>
	%4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
	%5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
	%6 = arith.muli %1, %cst_0 : tensor<2x1xi32>
	%7 = tt.broadcast %6 : tensor<2x1xi32> -> tensor<2x16xi32>
	%8 = tt.broadcast %5 : tensor<1x16xi32> -> tensor<2x16xi32>
	%9 = arith.addi %7, %8 : tensor<2x16xi32>
	%10 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%11 = tt.addptr %10, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%12 = tt.load %11 : tensor<2x16x!tt.ptr<i32>>
	%13 = arith.muli %3, %cst_0 : tensor<2x1xi32>
	%14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x1x!tt.ptr<i32>>
	%15 = tt.addptr %14, %13 : tensor<2x1x!tt.ptr<i32>>, tensor<2x1xi32>
	%16 = tt.broadcast %15 : tensor<2x1x!tt.ptr<i32>> -> tensor<2x16x!tt.ptr<i32>>
	%17 = tt.addptr %16, %8 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%18 = tt.atomic_rmw add, acq_rel, gpu, %17, %12, %cst : (tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>, tensor<2x16xi1>) -> tensor<2x16xi32>
	%19 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%20 = tt.addptr %19, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	tt.store %20, %18 : tensor<2x16x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func public @fully_unstructured_atomic_and_2d
// CHECK: hivm.hir.custom {extra_attr = "operate=and"
tt.func public @fully_unstructured_atomic_and_2d(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<4x4xi1>
	%cst_0 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_0 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x4x!tt.ptr<i64>>
	%8 = tt.addptr %7, %6 : tensor<4x4x!tt.ptr<i64>>, tensor<4x4xi32>
	%9 = tt.load %8 : tensor<4x4x!tt.ptr<i64>>
	%10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%11 = tt.addptr %10, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%12 = tt.load %11 : tensor<4x4x!tt.ptr<i32>>
	%13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%14 = tt.addptr %13, %9 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi64>
	%15 = tt.atomic_rmw and, acq_rel, gpu, %14, %12, %cst : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%16 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%17 = tt.addptr %16, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %17, %15 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// sizeInByte >= 64, falls back to scf.for loop unrolling + DiscreteMemAccess
// CHECK-LABEL: tt.func public @partial_structured_atomic_and_2d
// CHECK: scf.for
// CHECK: tt.atomic_rmw and, acq_rel, gpu, {{.*}} {DiscreteMemAccess}
// CHECK: {ExtractedLoadOrStore}
tt.func public @partial_structured_atomic_and_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<2x16xi1>
	%cst_0 = arith.constant dense<16> : tensor<2x1xi32>
	%cst_1 = arith.constant dense<2> : tensor<2x1xi32>
	%cst_2 = arith.constant dense<1> : tensor<2x1xi32>
	%0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
	%2 = arith.addi %1, %cst_2 : tensor<2x1xi32>
	%3 = arith.remsi %2, %cst_1 : tensor<2x1xi32>
	%4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
	%5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
	%6 = arith.muli %1, %cst_0 : tensor<2x1xi32>
	%7 = tt.broadcast %6 : tensor<2x1xi32> -> tensor<2x16xi32>
	%8 = tt.broadcast %5 : tensor<1x16xi32> -> tensor<2x16xi32>
	%9 = arith.addi %7, %8 : tensor<2x16xi32>
	%10 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%11 = tt.addptr %10, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%12 = tt.load %11 : tensor<2x16x!tt.ptr<i32>>
	%13 = arith.muli %3, %cst_0 : tensor<2x1xi32>
	%14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x1x!tt.ptr<i32>>
	%15 = tt.addptr %14, %13 : tensor<2x1x!tt.ptr<i32>>, tensor<2x1xi32>
	%16 = tt.broadcast %15 : tensor<2x1x!tt.ptr<i32>> -> tensor<2x16x!tt.ptr<i32>>
	%17 = tt.addptr %16, %8 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%18 = tt.atomic_rmw and, acq_rel, gpu, %17, %12, %cst : (tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>, tensor<2x16xi1>) -> tensor<2x16xi32>
	%19 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%20 = tt.addptr %19, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	tt.store %20, %18 : tensor<2x16x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func public @fully_unstructured_atomic_or_2d
// CHECK: hivm.hir.custom {extra_attr = "operate=or"
tt.func public @fully_unstructured_atomic_or_2d(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<4x4xi1>
	%cst_0 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_0 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x4x!tt.ptr<i64>>
	%8 = tt.addptr %7, %6 : tensor<4x4x!tt.ptr<i64>>, tensor<4x4xi32>
	%9 = tt.load %8 : tensor<4x4x!tt.ptr<i64>>
	%10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%11 = tt.addptr %10, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%12 = tt.load %11 : tensor<4x4x!tt.ptr<i32>>
	%13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%14 = tt.addptr %13, %9 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi64>
	%15 = tt.atomic_rmw or, acq_rel, gpu, %14, %12, %cst : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%16 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%17 = tt.addptr %16, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %17, %15 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// sizeInByte >= 64, falls back to scf.for loop unrolling + DiscreteMemAccess
// CHECK-LABEL: tt.func public @partial_structured_atomic_or_2d
// CHECK: scf.for
// CHECK: tt.atomic_rmw or, acq_rel, gpu, {{.*}} {DiscreteMemAccess}
tt.func public @partial_structured_atomic_or_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<2x16xi1>
	%cst_0 = arith.constant dense<16> : tensor<2x1xi32>
	%cst_1 = arith.constant dense<2> : tensor<2x1xi32>
	%cst_2 = arith.constant dense<1> : tensor<2x1xi32>
	%0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
	%2 = arith.addi %1, %cst_2 : tensor<2x1xi32>
	%3 = arith.remsi %2, %cst_1 : tensor<2x1xi32>
	%4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
	%5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
	%6 = arith.muli %1, %cst_0 : tensor<2x1xi32>
	%7 = tt.broadcast %6 : tensor<2x1xi32> -> tensor<2x16xi32>
	%8 = tt.broadcast %5 : tensor<1x16xi32> -> tensor<2x16xi32>
	%9 = arith.addi %7, %8 : tensor<2x16xi32>
	%10 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%11 = tt.addptr %10, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%12 = tt.load %11 : tensor<2x16x!tt.ptr<i32>>
	%13 = arith.muli %3, %cst_0 : tensor<2x1xi32>
	%14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x1x!tt.ptr<i32>>
	%15 = tt.addptr %14, %13 : tensor<2x1x!tt.ptr<i32>>, tensor<2x1xi32>
	%16 = tt.broadcast %15 : tensor<2x1x!tt.ptr<i32>> -> tensor<2x16x!tt.ptr<i32>>
	%17 = tt.addptr %16, %8 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%18 = tt.atomic_rmw or, acq_rel, gpu, %17, %12, %cst : (tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>, tensor<2x16xi1>) -> tensor<2x16xi32>
	%19 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%20 = tt.addptr %19, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	tt.store %20, %18 : tensor<2x16x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func public @fully_unstructured_atomic_xor_2d
// CHECK: hivm.hir.custom {extra_attr = "operate=xor"
tt.func public @fully_unstructured_atomic_xor_2d(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<4x4xi1>
	%cst_0 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_0 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x4x!tt.ptr<i64>>
	%8 = tt.addptr %7, %6 : tensor<4x4x!tt.ptr<i64>>, tensor<4x4xi32>
	%9 = tt.load %8 : tensor<4x4x!tt.ptr<i64>>
	%10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%11 = tt.addptr %10, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%12 = tt.load %11 : tensor<4x4x!tt.ptr<i32>>
	%13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%14 = tt.addptr %13, %9 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi64>
	%15 = tt.atomic_rmw xor, acq_rel, gpu, %14, %12, %cst : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%16 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%17 = tt.addptr %16, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %17, %15 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// sizeInByte >= 64, falls back to scf.for loop unrolling + DiscreteMemAccess
// CHECK-LABEL: tt.func public @partial_structured_atomic_xor_2d
// CHECK: scf.for
// CHECK: tt.atomic_rmw xor, acq_rel, gpu, {{.*}} {DiscreteMemAccess}
tt.func public @partial_structured_atomic_xor_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<2x16xi1>
	%cst_0 = arith.constant dense<16> : tensor<2x1xi32>
	%cst_1 = arith.constant dense<2> : tensor<2x1xi32>
	%cst_2 = arith.constant dense<1> : tensor<2x1xi32>
	%0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
	%2 = arith.addi %1, %cst_2 : tensor<2x1xi32>
	%3 = arith.remsi %2, %cst_1 : tensor<2x1xi32>
	%4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
	%5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
	%6 = arith.muli %1, %cst_0 : tensor<2x1xi32>
	%7 = tt.broadcast %6 : tensor<2x1xi32> -> tensor<2x16xi32>
	%8 = tt.broadcast %5 : tensor<1x16xi32> -> tensor<2x16xi32>
	%9 = arith.addi %7, %8 : tensor<2x16xi32>
	%10 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%11 = tt.addptr %10, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%12 = tt.load %11 : tensor<2x16x!tt.ptr<i32>>
	%13 = arith.muli %3, %cst_0 : tensor<2x1xi32>
	%14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x1x!tt.ptr<i32>>
	%15 = tt.addptr %14, %13 : tensor<2x1x!tt.ptr<i32>>, tensor<2x1xi32>
	%16 = tt.broadcast %15 : tensor<2x1x!tt.ptr<i32>> -> tensor<2x16x!tt.ptr<i32>>
	%17 = tt.addptr %16, %8 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%18 = tt.atomic_rmw xor, acq_rel, gpu, %17, %12, %cst : (tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>, tensor<2x16xi1>) -> tensor<2x16xi32>
	%19 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%20 = tt.addptr %19, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	tt.store %20, %18 : tensor<2x16x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func public @fully_unstructured_atomic_xchg_2d
// CHECK: hivm.hir.custom {extra_attr = "operate=xchg"
tt.func public @fully_unstructured_atomic_xchg_2d(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<4x4xi1>
	%cst_0 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_0 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x4x!tt.ptr<i64>>
	%8 = tt.addptr %7, %6 : tensor<4x4x!tt.ptr<i64>>, tensor<4x4xi32>
	%9 = tt.load %8 : tensor<4x4x!tt.ptr<i64>>
	%10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%11 = tt.addptr %10, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%12 = tt.load %11 : tensor<4x4x!tt.ptr<i32>>
	%13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%14 = tt.addptr %13, %9 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi64>
	%15 = tt.atomic_rmw exch, acq_rel, gpu, %14, %12, %cst : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%16 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%17 = tt.addptr %16, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %17, %15 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// sizeInByte >= 64, falls back to scf.for loop unrolling + DiscreteMemAccess
// CHECK-LABEL: tt.func public @partial_structured_atomic_xchg_2d
// CHECK: scf.for
// CHECK: tt.atomic_rmw exch, acq_rel, gpu, {{.*}} {DiscreteMemAccess}
tt.func public @partial_structured_atomic_xchg_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<2x16xi1>
	%cst_0 = arith.constant dense<16> : tensor<2x1xi32>
	%cst_1 = arith.constant dense<2> : tensor<2x1xi32>
	%cst_2 = arith.constant dense<1> : tensor<2x1xi32>
	%0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
	%2 = arith.addi %1, %cst_2 : tensor<2x1xi32>
	%3 = arith.remsi %2, %cst_1 : tensor<2x1xi32>
	%4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
	%5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
	%6 = arith.muli %1, %cst_0 : tensor<2x1xi32>
	%7 = tt.broadcast %6 : tensor<2x1xi32> -> tensor<2x16xi32>
	%8 = tt.broadcast %5 : tensor<1x16xi32> -> tensor<2x16xi32>
	%9 = arith.addi %7, %8 : tensor<2x16xi32>
	%10 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%11 = tt.addptr %10, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%12 = tt.load %11 : tensor<2x16x!tt.ptr<i32>>
	%13 = arith.muli %3, %cst_0 : tensor<2x1xi32>
	%14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x1x!tt.ptr<i32>>
	%15 = tt.addptr %14, %13 : tensor<2x1x!tt.ptr<i32>>, tensor<2x1xi32>
	%16 = tt.broadcast %15 : tensor<2x1x!tt.ptr<i32>> -> tensor<2x16x!tt.ptr<i32>>
	%17 = tt.addptr %16, %8 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%18 = tt.atomic_rmw exch, acq_rel, gpu, %17, %12, %cst : (tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>, tensor<2x16xi1>) -> tensor<2x16xi32>
	%19 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%20 = tt.addptr %19, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	tt.store %20, %18 : tensor<2x16x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func public @fully_unstructured_atomic_max_2d
// CHECK: hivm.hir.custom {extra_attr = "operate=max"
tt.func public @fully_unstructured_atomic_max_2d(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<4x4xi1>
	%cst_0 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_0 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x4x!tt.ptr<i64>>
	%8 = tt.addptr %7, %6 : tensor<4x4x!tt.ptr<i64>>, tensor<4x4xi32>
	%9 = tt.load %8 : tensor<4x4x!tt.ptr<i64>>
	%10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%11 = tt.addptr %10, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%12 = tt.load %11 : tensor<4x4x!tt.ptr<i32>>
	%13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%14 = tt.addptr %13, %9 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi64>
	%15 = tt.atomic_rmw max, acq_rel, gpu, %14, %12, %cst : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%16 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%17 = tt.addptr %16, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %17, %15 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// sizeInByte >= 64, falls back to scf.for loop unrolling + DiscreteMemAccess
// CHECK-LABEL: tt.func public @partial_structured_atomic_max_2d
// CHECK: scf.for
// CHECK: tt.atomic_rmw max, acq_rel, gpu, {{.*}} {DiscreteMemAccess}
tt.func public @partial_structured_atomic_max_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<2x16xi1>
	%cst_0 = arith.constant dense<16> : tensor<2x1xi32>
	%cst_1 = arith.constant dense<2> : tensor<2x1xi32>
	%cst_2 = arith.constant dense<1> : tensor<2x1xi32>
	%0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
	%2 = arith.addi %1, %cst_2 : tensor<2x1xi32>
	%3 = arith.remsi %2, %cst_1 : tensor<2x1xi32>
	%4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
	%5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
	%6 = arith.muli %1, %cst_0 : tensor<2x1xi32>
	%7 = tt.broadcast %6 : tensor<2x1xi32> -> tensor<2x16xi32>
	%8 = tt.broadcast %5 : tensor<1x16xi32> -> tensor<2x16xi32>
	%9 = arith.addi %7, %8 : tensor<2x16xi32>
	%10 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%11 = tt.addptr %10, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%12 = tt.load %11 : tensor<2x16x!tt.ptr<i32>>
	%13 = arith.muli %3, %cst_0 : tensor<2x1xi32>
	%14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x1x!tt.ptr<i32>>
	%15 = tt.addptr %14, %13 : tensor<2x1x!tt.ptr<i32>>, tensor<2x1xi32>
	%16 = tt.broadcast %15 : tensor<2x1x!tt.ptr<i32>> -> tensor<2x16x!tt.ptr<i32>>
	%17 = tt.addptr %16, %8 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%18 = tt.atomic_rmw max, acq_rel, gpu, %17, %12, %cst : (tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>, tensor<2x16xi1>) -> tensor<2x16xi32>
	%19 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%20 = tt.addptr %19, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	tt.store %20, %18 : tensor<2x16x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func public @fully_unstructured_atomic_min_2d
// CHECK: hivm.hir.custom {extra_attr = "operate=min"
tt.func public @fully_unstructured_atomic_min_2d(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<4x4xi1>
	%cst_0 = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst_0 : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x4x!tt.ptr<i64>>
	%8 = tt.addptr %7, %6 : tensor<4x4x!tt.ptr<i64>>, tensor<4x4xi32>
	%9 = tt.load %8 : tensor<4x4x!tt.ptr<i64>>
	%10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%11 = tt.addptr %10, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%12 = tt.load %11 : tensor<4x4x!tt.ptr<i32>>
	%13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%14 = tt.addptr %13, %9 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi64>
	%15 = tt.atomic_rmw min, acq_rel, gpu, %14, %12, %cst : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi1>) -> tensor<4x4xi32>
	%16 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%17 = tt.addptr %16, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %17, %15 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// sizeInByte >= 64, falls back to scf.for loop unrolling + DiscreteMemAccess
// CHECK-LABEL: tt.func public @partial_structured_atomic_min_2d
// CHECK: scf.for
// CHECK: tt.atomic_rmw min, acq_rel, gpu, {{.*}} {DiscreteMemAccess}
tt.func public @partial_structured_atomic_min_2d(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) {
	%cst = arith.constant dense<true> : tensor<2x16xi1>
	%cst_0 = arith.constant dense<16> : tensor<2x1xi32>
	%cst_1 = arith.constant dense<2> : tensor<2x1xi32>
	%cst_2 = arith.constant dense<1> : tensor<2x1xi32>
	%0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
	%2 = arith.addi %1, %cst_2 : tensor<2x1xi32>
	%3 = arith.remsi %2, %cst_1 : tensor<2x1xi32>
	%4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
	%5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
	%6 = arith.muli %1, %cst_0 : tensor<2x1xi32>
	%7 = tt.broadcast %6 : tensor<2x1xi32> -> tensor<2x16xi32>
	%8 = tt.broadcast %5 : tensor<1x16xi32> -> tensor<2x16xi32>
	%9 = arith.addi %7, %8 : tensor<2x16xi32>
	%10 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%11 = tt.addptr %10, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%12 = tt.load %11 : tensor<2x16x!tt.ptr<i32>>
	%13 = arith.muli %3, %cst_0 : tensor<2x1xi32>
	%14 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x1x!tt.ptr<i32>>
	%15 = tt.addptr %14, %13 : tensor<2x1x!tt.ptr<i32>>, tensor<2x1xi32>
	%16 = tt.broadcast %15 : tensor<2x1x!tt.ptr<i32>> -> tensor<2x16x!tt.ptr<i32>>
	%17 = tt.addptr %16, %8 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%18 = tt.atomic_rmw min, acq_rel, gpu, %17, %12, %cst : (tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>, tensor<2x16xi1>) -> tensor<2x16xi32>
	%19 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%20 = tt.addptr %19, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	tt.store %20, %18 : tensor<2x16x!tt.ptr<i32>>
	tt.return
}

// -----

// CHECK-LABEL: tt.func public @fully_unstructured_atomic_cas_2d
// CHECK: %[[OFFSET_FLAT:.*]] = tt.reshape {{.*}} : tensor<4x4xi64> -> tensor<16xi64>
// CHECK: %[[CMP_FLAT:.*]] = tt.reshape {{.*}} : tensor<4x4xi32> -> tensor<16xi32>
// CHECK: %[[VAL_FLAT:.*]] = tt.reshape {{.*}} : tensor<4x4xi32> -> tensor<16xi32>
// CHECK: hivm.hir.custom {extra_attr = "operate=cas"
tt.func public @fully_unstructured_atomic_cas_2d(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>, %arg3: !tt.ptr<i32>, %arg4: !tt.ptr<i32>) attributes {noinline = false} {
	%cst = arith.constant dense<4> : tensor<4x1xi32>
	%0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
	%2 = tt.expand_dims %0 {axis = 0 : i32} : tensor<4xi32> -> tensor<1x4xi32>
	%3 = arith.muli %1, %cst : tensor<4x1xi32>
	%4 = tt.broadcast %3 : tensor<4x1xi32> -> tensor<4x4xi32>
	%5 = tt.broadcast %2 : tensor<1x4xi32> -> tensor<4x4xi32>
	%6 = arith.addi %4, %5 : tensor<4x4xi32>
	%7 = tt.splat %arg0 : !tt.ptr<i64> -> tensor<4x4x!tt.ptr<i64>>
	%8 = tt.addptr %7, %6 : tensor<4x4x!tt.ptr<i64>>, tensor<4x4xi32>
	%9 = tt.load %8 : tensor<4x4x!tt.ptr<i64>>
	%10 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%11 = tt.addptr %10, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%12 = tt.load %11 : tensor<4x4x!tt.ptr<i32>>
	%13 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%14 = tt.addptr %13, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	%15 = tt.load %14 : tensor<4x4x!tt.ptr<i32>>
	%16 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%17 = tt.addptr %16, %9 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi64>
	%18 = tt.atomic_cas acq_rel, gpu, %17, %12, %15 : (tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
	%19 = tt.splat %arg4 : !tt.ptr<i32> -> tensor<4x4x!tt.ptr<i32>>
	%20 = tt.addptr %19, %6 : tensor<4x4x!tt.ptr<i32>>, tensor<4x4xi32>
	tt.store %20, %18 : tensor<4x4x!tt.ptr<i32>>
	tt.return
}

// -----

// sizeInByte >= 64, falls back to scf.for loop unrolling + DiscreteMemAccess
// CHECK-LABEL: tt.func public @partial_structured_atomic_cas_2d
// CHECK: scf.for
// CHECK: tt.atomic_cas acq_rel, gpu, {{.*}} {DiscreteMemAccess}
// CHECK: {ExtractedLoadOrStore}
tt.func public @partial_structured_atomic_cas_2d(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
	%cst = arith.constant dense<16> : tensor<2x1xi32>
	%cst_0 = arith.constant dense<2> : tensor<2x1xi32>
	%cst_1 = arith.constant dense<1> : tensor<2x1xi32>
	%0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
	%1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
	%2 = arith.addi %1, %cst_1 : tensor<2x1xi32>
	%3 = arith.remsi %2, %cst_0 : tensor<2x1xi32>
	%4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
	%5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
	%6 = arith.muli %1, %cst : tensor<2x1xi32>
	%7 = tt.broadcast %6 : tensor<2x1xi32> -> tensor<2x16xi32>
	%8 = tt.broadcast %5 : tensor<1x16xi32> -> tensor<2x16xi32>
	%9 = arith.addi %7, %8 : tensor<2x16xi32>
	%10 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%11 = tt.addptr %10, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%12 = tt.load %11 : tensor<2x16x!tt.ptr<i32>>
	%13 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%14 = tt.addptr %13, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%15 = tt.load %14 : tensor<2x16x!tt.ptr<i32>>
	%16 = arith.muli %3, %cst : tensor<2x1xi32>
	%17 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<2x1x!tt.ptr<i32>>
	%18 = tt.addptr %17, %16 : tensor<2x1x!tt.ptr<i32>>, tensor<2x1xi32>
	%19 = tt.broadcast %18 : tensor<2x1x!tt.ptr<i32>> -> tensor<2x16x!tt.ptr<i32>>
	%20 = tt.addptr %19, %8 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	%21 = tt.atomic_cas acq_rel, gpu, %20, %12, %15 : (tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>, tensor<2x16xi32>) -> tensor<2x16xi32>
	%22 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
	%23 = tt.addptr %22, %9 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
	tt.store %23, %21 : tensor<2x16x!tt.ptr<i32>>
	tt.return
}
