// RUN: triton-opt %s '--triton-to-unstructure=compile-on-910-95=True compile-mode=simd_simt_template' | FileCheck %s

// CHECK-LABEL: tt.func @triton_indirect_load
// CHECK: ascend.indirect_load {{.*}} -> tensor<1024xi32>
tt.func @triton_indirect_load(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst = arith.constant dense<0> : tensor<1024xi32>
  %cst_0 = arith.constant dense<200> : tensor<1024xi32>
  %cst_1 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst_0 : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_1 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %5, %3, %cst {MixCompileDiscreteMask} : tensor<1024x!tt.ptr<i32>>
  tt.store %7, %8 : tensor<1024x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @triton_indirect_store
// CHECK: ascend.indirect_store {{.*}}
tt.func @triton_indirect_store(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<i32>) {
  %cst_0 = arith.constant dense<200> : tensor<1024xi32>
  %cst_1 = arith.constant dense<400> : tensor<1024xi32>
  %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
  %1 = arith.cmpi slt, %0, %cst_0 : tensor<1024xi32>
  %2 = arith.cmpi sgt, %0, %cst_1 : tensor<1024xi32>
  %3 = arith.ori %1, %2 : tensor<1024xi1>
  %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %5 = tt.addptr %4, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
  %7 = tt.addptr %6, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>
  %8 = tt.load %5 : tensor<1024x!tt.ptr<i32>>
  tt.store %7, %8, %3 {MixCompileDiscreteMask} : tensor<1024x!tt.ptr<i32>>
  tt.return
}

// CHECK-LABEL: tt.func @discrete_highrank_and_structured_lowrank_loadstore_2d
// CHECK: ascend.indirect_load {{.*}} -> tensor<2x16xbf16>
// CHECK: ascend.indirect_store {{.*}}
tt.func @discrete_highrank_and_structured_lowrank_loadstore_2d(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<2.000000e+00> : tensor<2x16xbf16>
  %cst_0 = arith.constant dense<16> : tensor<2x1xi32>
  %cst_1 = arith.constant dense<2> : tensor<2x1xi32>
  %cst_2 = arith.constant dense<1> : tensor<2x1xi32>
  %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %2 = arith.addi %1, %cst_2 : tensor<2x1xi32>
  %3 = arith.remsi %2, %cst_1 : tensor<2x1xi32>
  %4 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %6 = arith.muli %3, %cst_0 : tensor<2x1xi32>
  %7 = tt.broadcast %6 : tensor<2x1xi32> -> tensor<2x16xi32>
  %8 = tt.broadcast %5 : tensor<1x16xi32> -> tensor<2x16xi32>
  %9 = arith.addi %7, %8 : tensor<2x16xi32>
  %10 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<2x16x!tt.ptr<bf16>>
  %11 = tt.addptr %10, %9 : tensor<2x16x!tt.ptr<bf16>>, tensor<2x16xi32>
  %12 = tt.load %11 : tensor<2x16x!tt.ptr<bf16>>
  %13 = arith.addf %12, %cst : tensor<2x16xbf16>
  tt.store %11, %13 : tensor<2x16x!tt.ptr<bf16>>
  tt.return
}

// CHECK-LABEL: tt.func @discrete_highrank_and_structured_lowrank_loadstore_3d
// CHECK: ascend.indirect_load {{.*}} -> tensor<2x2x8xbf16>
// CHECK: ascend.indirect_store {{.*}}
tt.func @discrete_highrank_and_structured_lowrank_loadstore_3d(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<3.000000e+00> : tensor<2x2x8xbf16>
  %cst_0 = arith.constant dense<8> : tensor<2x2x1xi32>
  %cst_1 = arith.constant dense<2> : tensor<2x1xi32>
  %cst_2 = arith.constant dense<1> : tensor<2x1xi32>
  %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %2 = arith.addi %1, %cst_2 : tensor<2x1xi32>
  %3 = arith.remsi %2, %cst_1 : tensor<2x1xi32>
  %4 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %5 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %6 = tt.expand_dims %5 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %7 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32>
  %8 = arith.muli %3, %cst_1 : tensor<2x1xi32>
  %9 = tt.broadcast %8 : tensor<2x1xi32> -> tensor<2x2xi32>
  %10 = tt.broadcast %4 : tensor<1x2xi32> -> tensor<2x2xi32>
  %11 = arith.addi %9, %10 : tensor<2x2xi32>
  %12 = tt.expand_dims %11 {axis = 2 : i32} : tensor<2x2xi32> -> tensor<2x2x1xi32>
  %13 = arith.muli %12, %cst_0 : tensor<2x2x1xi32>
  %14 = tt.broadcast %13 : tensor<2x2x1xi32> -> tensor<2x2x8xi32>
  %15 = tt.broadcast %7 : tensor<1x1x8xi32> -> tensor<2x2x8xi32>
  %16 = arith.addi %14, %15 : tensor<2x2x8xi32>
  %17 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<2x2x8x!tt.ptr<bf16>>
  %18 = tt.addptr %17, %16 : tensor<2x2x8x!tt.ptr<bf16>>, tensor<2x2x8xi32>
  %19 = tt.load %18 : tensor<2x2x8x!tt.ptr<bf16>>
  %20 = arith.addf %19, %cst : tensor<2x2x8xbf16>
  tt.store %18, %20 : tensor<2x2x8x!tt.ptr<bf16>>
  tt.return
}

// CHECK-LABEL: tt.func @discrete_highrank_and_structured_lowrank_loadstore_4d
// CHECK: %[[VAL_0:.*]] = scf.for %[[VAL_1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[VAL_2:.*]] = %{{.*}}) -> (tensor<2x2x2x8xbf16>) {
// CHECK:   %[[VAL_3:.*]] = tensor.extract_slice %{{.*}}{{\[}}%[[VAL_1]], 0, 0, 0] [1, 2, 2, 8] [1, 1, 1, 1] {DiscreteMemAccess} : tensor<2x2x2x8xi64> to tensor<1x2x2x8xi64>
// CHECK:   %[[VAL_4:.*]] = tt.splat %{{.*}} : !tt.ptr<bf16> -> tensor<1x2x2x8x!tt.ptr<bf16>>
// CHECK:   %[[VAL_5:.*]] = tt.addptr %[[VAL_4]], %[[VAL_3]] : tensor<1x2x2x8x!tt.ptr<bf16>>, tensor<1x2x2x8xi64>
// CHECK:   %[[VAL_6:.*]] = tt.load %[[VAL_5]] {DiscreteMemAccess} : tensor<1x2x2x8x!tt.ptr<bf16>>
// CHECK:   %[[VAL_7:.*]] = tensor.insert_slice %[[VAL_6]] into %[[VAL_2]]{{\[}}%[[VAL_1]], 0, 0, 0] [1, 2, 2, 8] [1, 1, 1, 1] : tensor<1x2x2x8xbf16> into tensor<2x2x2x8xbf16>
// CHECK:   scf.yield {DiscreteMemAccess} %[[VAL_7]] : tensor<2x2x2x8xbf16>
// CHECK: } {ExtractedLoadOrStore}
// CHECK: %[[VAL_8:.*]] = arith.addf %[[VAL_0]], %{{.*}} : tensor<2x2x2x8xbf16>
// CHECK: scf.for %[[VAL_9:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[VAL_10:.*]] = tensor.extract_slice %{{.*}}{{\[}}%[[VAL_9]], 0, 0, 0] [1, 2, 2, 8] [1, 1, 1, 1] {DiscreteMemAccess} : tensor<2x2x2x8xi64> to tensor<1x2x2x8xi64>
// CHECK:   %[[VAL_11:.*]] = tt.splat %{{.*}} : !tt.ptr<bf16> -> tensor<1x2x2x8x!tt.ptr<bf16>>
// CHECK:   %[[VAL_12:.*]] = tt.addptr %[[VAL_11]], %[[VAL_10]] : tensor<1x2x2x8x!tt.ptr<bf16>>, tensor<1x2x2x8xi64>
// CHECK:   %[[VAL_13:.*]] = tensor.extract_slice %[[VAL_8]]{{\[}}%[[VAL_9]], 0, 0, 0] [1, 2, 2, 8] [1, 1, 1, 1] {DiscreteMemAccess} : tensor<2x2x2x8xbf16> to tensor<1x2x2x8xbf16>
// CHECK:   tt.store %[[VAL_12]], %[[VAL_13]] {DiscreteMemAccess} : tensor<1x2x2x8x!tt.ptr<bf16>>
// CHECK: } {ExtractedLoadOrStore}
tt.func @discrete_highrank_and_structured_lowrank_loadstore_4d(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<4.000000e+00> : tensor<2x2x2x8xbf16>
  %cst_0 = arith.constant dense<8> : tensor<2x2x2x1xi32>
  %cst_1 = arith.constant dense<2> : tensor<2x2x1xi32>
  %cst_2 = arith.constant dense<2> : tensor<2x1x1xi32>
  %cst_3 = arith.constant dense<1> : tensor<2x1x1xi32>
  %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %2 = tt.expand_dims %1 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
  %3 = arith.addi %2, %cst_3 : tensor<2x1x1xi32>
  %4 = arith.remsi %3, %cst_2 : tensor<2x1x1xi32>
  %5 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %6 = tt.expand_dims %5 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32>
  %7 = tt.expand_dims %5 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32>
  %8 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %10 = tt.expand_dims %9 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32>
  %11 = tt.expand_dims %10 {axis = 2 : i32} : tensor<1x1x8xi32> -> tensor<1x1x1x8xi32>
  %12 = arith.muli %4, %cst_2 : tensor<2x1x1xi32>
  %13 = tt.broadcast %12 : tensor<2x1x1xi32> -> tensor<2x2x1xi32>
  %14 = tt.broadcast %6 : tensor<1x2x1xi32> -> tensor<2x2x1xi32>
  %15 = arith.addi %13, %14 : tensor<2x2x1xi32>
  %16 = arith.muli %15, %cst_1 : tensor<2x2x1xi32>
  %17 = tt.broadcast %16 : tensor<2x2x1xi32> -> tensor<2x2x2xi32>
  %18 = tt.broadcast %7 : tensor<1x1x2xi32> -> tensor<2x2x2xi32>
  %19 = arith.addi %17, %18 : tensor<2x2x2xi32>
  %20 = tt.expand_dims %19 {axis = 3 : i32} : tensor<2x2x2xi32> -> tensor<2x2x2x1xi32>
  %21 = arith.muli %20, %cst_0 : tensor<2x2x2x1xi32>
  %22 = tt.broadcast %21 : tensor<2x2x2x1xi32> -> tensor<2x2x2x8xi32>
  %23 = tt.broadcast %11 : tensor<1x1x1x8xi32> -> tensor<2x2x2x8xi32>
  %24 = arith.addi %22, %23 : tensor<2x2x2x8xi32>
  %25 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<2x2x2x8x!tt.ptr<bf16>>
  %26 = tt.addptr %25, %24 : tensor<2x2x2x8x!tt.ptr<bf16>>, tensor<2x2x2x8xi32>
  %27 = tt.load %26 : tensor<2x2x2x8x!tt.ptr<bf16>>
  %28 = arith.addf %27, %cst : tensor<2x2x2x8xbf16>
  tt.store %26, %28 : tensor<2x2x2x8x!tt.ptr<bf16>>
  tt.return
}

// CHECK-LABEL: tt.func @discrete_highrank_and_structured_lowrank_loadstore_5d
// CHECK: %[[VAL_0:.*]] = scf.for %[[VAL_1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[VAL_2:.*]] = %{{.*}}) -> (tensor<2x2x2x2x8xbf16>) {
// CHECK:   %[[VAL_3:.*]] = tensor.extract_slice %{{.*}}{{\[}}%[[VAL_1]], 0, 0, 0, 0] [1, 2, 2, 2, 8] [1, 1, 1, 1, 1] {DiscreteMemAccess} : tensor<2x2x2x2x8xi64> to tensor<1x2x2x2x8xi64>
// CHECK:   %[[VAL_4:.*]] = tt.splat %{{.*}} : !tt.ptr<bf16> -> tensor<1x2x2x2x8x!tt.ptr<bf16>>
// CHECK:   %[[VAL_5:.*]] = tt.addptr %[[VAL_4]], %[[VAL_3]] : tensor<1x2x2x2x8x!tt.ptr<bf16>>, tensor<1x2x2x2x8xi64>
// CHECK:   %[[VAL_6:.*]] = tt.load %[[VAL_5]] {DiscreteMemAccess} : tensor<1x2x2x2x8x!tt.ptr<bf16>>
// CHECK:   %[[VAL_7:.*]] = tensor.insert_slice %[[VAL_6]] into %[[VAL_2]]{{\[}}%[[VAL_1]], 0, 0, 0, 0] [1, 2, 2, 2, 8] [1, 1, 1, 1, 1] : tensor<1x2x2x2x8xbf16> into tensor<2x2x2x2x8xbf16>
// CHECK:   scf.yield {DiscreteMemAccess} %[[VAL_7]] : tensor<2x2x2x2x8xbf16>
// CHECK: } {ExtractedLoadOrStore}
// CHECK: %[[VAL_8:.*]] = arith.addf %[[VAL_0]], %{{.*}} : tensor<2x2x2x2x8xbf16>
// CHECK: scf.for %[[VAL_9:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[VAL_10:.*]] = tensor.extract_slice %{{.*}}{{\[}}%[[VAL_9]], 0, 0, 0, 0] [1, 2, 2, 2, 8] [1, 1, 1, 1, 1] {DiscreteMemAccess} : tensor<2x2x2x2x8xi64> to tensor<1x2x2x2x8xi64>
// CHECK:   %[[VAL_11:.*]] = tt.splat %{{.*}} : !tt.ptr<bf16> -> tensor<1x2x2x2x8x!tt.ptr<bf16>>
// CHECK:   %[[VAL_12:.*]] = tt.addptr %[[VAL_11]], %[[VAL_10]] : tensor<1x2x2x2x8x!tt.ptr<bf16>>, tensor<1x2x2x2x8xi64>
// CHECK:   %[[VAL_13:.*]] = tensor.extract_slice %[[VAL_8]]{{\[}}%[[VAL_9]], 0, 0, 0, 0] [1, 2, 2, 2, 8] [1, 1, 1, 1, 1] {DiscreteMemAccess} : tensor<2x2x2x2x8xbf16> to tensor<1x2x2x2x8xbf16>
// CHECK:   tt.store %[[VAL_12]], %[[VAL_13]] {DiscreteMemAccess} : tensor<1x2x2x2x8x!tt.ptr<bf16>>
// CHECK: } {ExtractedLoadOrStore}
tt.func @discrete_highrank_and_structured_lowrank_loadstore_5d(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<5.000000e+00> : tensor<2x2x2x2x8xbf16>
  %cst_0 = arith.constant dense<8> : tensor<2x2x2x2x1xi32>
  %cst_1 = arith.constant dense<2> : tensor<2x2x2x1xi32>
  %cst_2 = arith.constant dense<2> : tensor<2x2x1x1xi32>
  %cst_3 = arith.constant dense<2> : tensor<2x1x1x1xi32>
  %cst_4 = arith.constant dense<1> : tensor<2x1x1x1xi32>
  %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
  %2 = tt.expand_dims %1 {axis = 2 : i32} : tensor<2x1xi32> -> tensor<2x1x1xi32>
  %3 = tt.expand_dims %2 {axis = 3 : i32} : tensor<2x1x1xi32> -> tensor<2x1x1x1xi32>
  %4 = arith.addi %3, %cst_4 : tensor<2x1x1x1xi32>
  %5 = arith.remsi %4, %cst_3 : tensor<2x1x1x1xi32>
  %6 = tt.expand_dims %0 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
  %7 = tt.expand_dims %6 {axis = 2 : i32} : tensor<1x2xi32> -> tensor<1x2x1xi32>
  %8 = tt.expand_dims %7 {axis = 3 : i32} : tensor<1x2x1xi32> -> tensor<1x2x1x1xi32>
  %9 = tt.expand_dims %6 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32>
  %10 = tt.expand_dims %9 {axis = 3 : i32} : tensor<1x1x2xi32> -> tensor<1x1x2x1xi32>
  %11 = tt.expand_dims %9 {axis = 2 : i32} : tensor<1x1x2xi32> -> tensor<1x1x1x2xi32>
  %12 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %13 = tt.expand_dims %12 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x8xi32> -> tensor<1x1x8xi32>
  %15 = tt.expand_dims %14 {axis = 2 : i32} : tensor<1x1x8xi32> -> tensor<1x1x1x8xi32>
  %16 = tt.expand_dims %15 {axis = 3 : i32} : tensor<1x1x1x8xi32> -> tensor<1x1x1x1x8xi32>
  %17 = arith.muli %5, %cst_3 : tensor<2x1x1x1xi32>
  %18 = tt.broadcast %17 : tensor<2x1x1x1xi32> -> tensor<2x2x1x1xi32>
  %19 = tt.broadcast %8 : tensor<1x2x1x1xi32> -> tensor<2x2x1x1xi32>
  %20 = arith.addi %18, %19 : tensor<2x2x1x1xi32>
  %21 = arith.muli %20, %cst_2 : tensor<2x2x1x1xi32>
  %22 = tt.broadcast %21 : tensor<2x2x1x1xi32> -> tensor<2x2x2x1xi32>
  %23 = tt.broadcast %10 : tensor<1x1x2x1xi32> -> tensor<2x2x2x1xi32>
  %24 = arith.addi %22, %23 : tensor<2x2x2x1xi32>
  %25 = arith.muli %24, %cst_1 : tensor<2x2x2x1xi32>
  %26 = tt.broadcast %25 : tensor<2x2x2x1xi32> -> tensor<2x2x2x2xi32>
  %27 = tt.broadcast %11 : tensor<1x1x1x2xi32> -> tensor<2x2x2x2xi32>
  %28 = arith.addi %26, %27 : tensor<2x2x2x2xi32>
  %29 = tt.expand_dims %28 {axis = 4 : i32} : tensor<2x2x2x2xi32> -> tensor<2x2x2x2x1xi32>
  %30 = arith.muli %29, %cst_0 : tensor<2x2x2x2x1xi32>
  %31 = tt.broadcast %30 : tensor<2x2x2x2x1xi32> -> tensor<2x2x2x2x8xi32>
  %32 = tt.broadcast %16 : tensor<1x1x1x1x8xi32> -> tensor<2x2x2x2x8xi32>
  %33 = arith.addi %31, %32 : tensor<2x2x2x2x8xi32>
  %34 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<2x2x2x2x8x!tt.ptr<bf16>>
  %35 = tt.addptr %34, %33 : tensor<2x2x2x2x8x!tt.ptr<bf16>>, tensor<2x2x2x2x8xi32>
  %36 = tt.load %35 : tensor<2x2x2x2x8x!tt.ptr<bf16>>
  %37 = arith.addf %36, %cst : tensor<2x2x2x2x8xbf16>
  tt.store %35, %37 : tensor<2x2x2x2x8x!tt.ptr<bf16>>
  tt.return
}
