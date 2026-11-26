// RUN: triton-opt '--triton-to-linalg=global-kernel=False named-ops=True enable-nd2nz-on-vector=False enable-select-analysis=True compile-on-910-95=True' --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @addptr_implicit_perm_load_2d(
// CHECK: %[[IN_STRIDE0:[A-Za-z0-9_]+]] = arith.index_cast %arg6 : i32 to index
// CHECK: %[[IN_CAST0:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg2 to offset: [%[[IN_OFF0:[A-Za-z0-9_]+]]], sizes: [4, 64], strides: [1, %[[IN_STRIDE0]]] : memref<?xf32> to memref<4x64xf32, strided<[1, ?], offset: ?>>
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<?x?xf32, strided<[1, ?], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
// CHECK: %[[TENSOR0:[A-Za-z0-9_]+]] = bufferization.to_tensor %{{.*}} restrict writable : memref<4x64xf32>
// CHECK: %[[OUT_STRIDE0:[A-Za-z0-9_]+]] = arith.index_cast %arg7 : i32 to index
// CHECK: %[[OUT_CAST0:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg3 to offset: [%[[OUT_OFF0:[A-Za-z0-9_]+]]], sizes: [4, 64], strides: [%[[OUT_STRIDE0]], 1] : memref<?xf32> to memref<4x64xf32, strided<[?, 1], offset: ?>>
// CHECK: %[[SLICE0:[A-Za-z0-9_]+]] = tensor.extract_slice %[[TENSOR0]][0, 0] [%[[ROWS0:[A-Za-z0-9_]+]], %[[COLS0:[A-Za-z0-9_]+]]] [1, 1] : tensor<4x64xf32> to tensor<?x?xf32>
// CHECK-NOT: linalg.transpose

// CHECK-LABEL: func.func @addptr_implicit_perm_store_2d(
// CHECK: %[[IN_STRIDE1:[A-Za-z0-9_]+]] = arith.index_cast %arg6 : i32 to index
// CHECK: %[[IN_CAST1:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg2 to offset: [%[[IN_OFF1:[A-Za-z0-9_]+]]], sizes: [4, 64], strides: [%[[IN_STRIDE1]], 1] : memref<?xf32> to memref<4x64xf32, strided<[?, 1], offset: ?>>
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
// CHECK: %[[TENSOR1:[A-Za-z0-9_]+]] = bufferization.to_tensor %{{.*}} restrict writable : memref<4x64xf32>
// CHECK: %[[OUT_STRIDE1:[A-Za-z0-9_]+]] = arith.index_cast %arg7 : i32 to index
// CHECK: %[[OUT_CAST1:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg3 to offset: [%[[OUT_OFF1:[A-Za-z0-9_]+]]], sizes: [64, 4], strides: [%[[OUT_STRIDE1]], 1] : memref<?xf32> to memref<64x4xf32, strided<[?, 1], offset: ?>>
// CHECK: %[[EMPTY1:[A-Za-z0-9_]+]] = tensor.empty() : tensor<64x4xf32>
// CHECK-NEXT: %[[TRANS1:[A-Za-z0-9_]+]] = linalg.transpose ins(%[[TENSOR1]] : tensor<4x64xf32>) outs(%[[EMPTY1]] : tensor<64x4xf32>) permutation = [1, 0]
// CHECK: %[[SLICE1:[A-Za-z0-9_]+]] = tensor.extract_slice %[[TRANS1]][0, 0] [%[[COLS1:[A-Za-z0-9_]+]], %[[ROWS1:[A-Za-z0-9_]+]]] [1, 1] : tensor<64x4xf32> to tensor<?x?xf32>

// CHECK-LABEL: func.func @make_tensor_ptr_implicit_perm_load_2d(
// CHECK: %[[IN_STRIDE2:[A-Za-z0-9_]+]] = arith.index_cast %arg6 : i32 to index
// CHECK: %[[IN_CAST2:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg2 to offset: [%[[IN_OFF2:[A-Za-z0-9_]+]]], sizes: [4, 64], strides: [1, %[[IN_STRIDE2]]] : memref<?xf32> to memref<4x64xf32, strided<[1, ?], offset: ?>>
// CHECK: memref.copy %[[IN_CAST2]], %{{.*}} : memref<4x64xf32, strided<[1, ?], offset: ?>> to memref<4x64xf32>
// CHECK: %[[TENSOR2:[A-Za-z0-9_]+]] = bufferization.to_tensor %{{.*}} restrict writable : memref<4x64xf32>
// CHECK: %[[OUT_STRIDE2:[A-Za-z0-9_]+]] = arith.index_cast %arg7 : i32 to index
// CHECK: %[[OUT_CAST2:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg3 to offset: [%[[OUT_OFF2:[A-Za-z0-9_]+]]], sizes: [4, 64], strides: [%[[OUT_STRIDE2]], 1] : memref<?xf32> to memref<4x64xf32, strided<[?, 1], offset: ?>>
// CHECK: %[[SLICE2:[A-Za-z0-9_]+]] = tensor.extract_slice %[[TENSOR2]][0, 0] [%[[ROWS2:[A-Za-z0-9_]+]], %[[COLS2:[A-Za-z0-9_]+]]] [1, 1] : tensor<4x64xf32> to tensor<?x?xf32>
// CHECK-NOT: linalg.transpose

// CHECK-LABEL: func.func @make_tensor_ptr_implicit_perm_store_2d(
// CHECK: %[[IN_STRIDE3:[A-Za-z0-9_]+]] = arith.index_cast %arg6 : i32 to index
// CHECK: %[[IN_CAST3:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg2 to offset: [%[[IN_OFF3:[A-Za-z0-9_]+]]], sizes: [4, 64], strides: [%[[IN_STRIDE3]], 1] : memref<?xf32> to memref<4x64xf32, strided<[?, 1], offset: ?>>
// CHECK: memref.copy %{{.*}}, %{{.*}} : memref<?x?xf32, strided<[?, 1], offset: ?>> to memref<?x?xf32, strided<[64, 1]>>
// CHECK: %[[TENSOR3:[A-Za-z0-9_]+]] = bufferization.to_tensor %{{.*}} restrict writable : memref<4x64xf32>
// CHECK: %[[OUT_STRIDE3:[A-Za-z0-9_]+]] = arith.index_cast %arg7 : i32 to index
// CHECK: %[[OUT_CAST3:[A-Za-z0-9_]+]] = memref.reinterpret_cast %arg3 to offset: [%[[OUT_OFF3:[A-Za-z0-9_]+]]], sizes: [64, 4], strides: [%[[OUT_STRIDE3]], 1] : memref<?xf32> to memref<64x4xf32, strided<[?, 1], offset: ?>>
// CHECK: %[[EMPTY3:[A-Za-z0-9_]+]] = tensor.empty() : tensor<64x4xf32>
// CHECK-NEXT: %[[TRANS3:[A-Za-z0-9_]+]] = linalg.transpose ins(%[[TENSOR3]] : tensor<4x64xf32>) outs(%[[EMPTY3]] : tensor<64x4xf32>) permutation = [1, 0]
// CHECK: %[[SLICE3:[A-Za-z0-9_]+]] = tensor.extract_slice %[[TRANS3]]{{.*}} : tensor<64x4xf32> to tensor<?x?xf32>

// addptr load implicit permutation
module {
  tt.func public @addptr_implicit_perm_load_2d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x64xf32>
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %4 = tt.splat %1 : i32 -> tensor<4x1xi32>
    %5 = arith.addi %4, %3 : tensor<4x1xi32>
    %6 = tt.get_program_id y : i32
    %7 = arith.muli %6, %c64_i32 : i32
    %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %10 = tt.splat %7 : i32 -> tensor<1x64xi32>
    %11 = arith.addi %10, %9 : tensor<1x64xi32>
    %12 = tt.splat %arg3 : i32 -> tensor<4x1xi32>
    %13 = arith.cmpi slt, %5, %12 : tensor<4x1xi32>
    %14 = tt.splat %arg2 : i32 -> tensor<1x64xi32>
    %15 = arith.cmpi slt, %11, %14 : tensor<1x64xi32>
    %16 = tt.broadcast %13 : tensor<4x1xi1> -> tensor<4x64xi1>
    %17 = tt.broadcast %15 : tensor<1x64xi1> -> tensor<4x64xi1>
    %18 = arith.andi %16, %17 : tensor<4x64xi1>
    %19 = tt.splat %arg4 : i32 -> tensor<1x64xi32>
    %20 = arith.muli %11, %19 : tensor<1x64xi32>
    %21 = tt.broadcast %5 : tensor<4x1xi32> -> tensor<4x64xi32>
    %22 = tt.broadcast %20 : tensor<1x64xi32> -> tensor<4x64xi32>
    %23 = arith.addi %21, %22 : tensor<4x64xi32>
    %24 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x64x!tt.ptr<f32>>
    %25 = tt.addptr %24, %23 : tensor<4x64x!tt.ptr<f32>>, tensor<4x64xi32>
    %26 = tt.load %25, %18, %cst : tensor<4x64x!tt.ptr<f32>>
    %27 = tt.splat %arg5 : i32 -> tensor<4x1xi32>
    %28 = arith.muli %5, %27 : tensor<4x1xi32>
    %29 = tt.broadcast %28 : tensor<4x1xi32> -> tensor<4x64xi32>
    %30 = tt.broadcast %11 : tensor<1x64xi32> -> tensor<4x64xi32>
    %31 = arith.addi %29, %30 : tensor<4x64xi32>
    %32 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x64x!tt.ptr<f32>>
    %33 = tt.addptr %32, %31 : tensor<4x64x!tt.ptr<f32>>, tensor<4x64xi32>
    tt.store %33, %26, %18 : tensor<4x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// addptr store implicit permutation
module {
  tt.func public @addptr_implicit_perm_store_2d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x64xf32>
    %c64_i32 = arith.constant 64 : i32
    %c4_i32 = arith.constant 4 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c4_i32 : i32
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %4 = tt.splat %1 : i32 -> tensor<4x1xi32>
    %5 = arith.addi %4, %3 : tensor<4x1xi32>
    %6 = tt.get_program_id y : i32
    %7 = arith.muli %6, %c64_i32 : i32
    %8 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %9 = tt.expand_dims %8 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %10 = tt.splat %7 : i32 -> tensor<1x64xi32>
    %11 = arith.addi %10, %9 : tensor<1x64xi32>
    %12 = tt.splat %arg3 : i32 -> tensor<4x1xi32>
    %13 = arith.cmpi slt, %5, %12 : tensor<4x1xi32>
    %14 = tt.splat %arg2 : i32 -> tensor<1x64xi32>
    %15 = arith.cmpi slt, %11, %14 : tensor<1x64xi32>
    %16 = tt.broadcast %13 : tensor<4x1xi1> -> tensor<4x64xi1>
    %17 = tt.broadcast %15 : tensor<1x64xi1> -> tensor<4x64xi1>
    %18 = arith.andi %16, %17 : tensor<4x64xi1>
    %19 = tt.splat %arg4 : i32 -> tensor<4x1xi32>
    %20 = arith.muli %5, %19 : tensor<4x1xi32>
    %21 = tt.broadcast %20 : tensor<4x1xi32> -> tensor<4x64xi32>
    %22 = tt.broadcast %11 : tensor<1x64xi32> -> tensor<4x64xi32>
    %23 = arith.addi %21, %22 : tensor<4x64xi32>
    %24 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x64x!tt.ptr<f32>>
    %25 = tt.addptr %24, %23 : tensor<4x64x!tt.ptr<f32>>, tensor<4x64xi32>
    %26 = tt.load %25, %18, %cst : tensor<4x64x!tt.ptr<f32>>
    %27 = tt.splat %arg5 : i32 -> tensor<1x64xi32>
    %28 = arith.muli %11, %27 : tensor<1x64xi32>
    %29 = tt.broadcast %5 : tensor<4x1xi32> -> tensor<4x64xi32>
    %30 = tt.broadcast %28 : tensor<1x64xi32> -> tensor<4x64xi32>
    %31 = arith.addi %29, %30 : tensor<4x64xi32>
    %32 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x64x!tt.ptr<f32>>
    %33 = tt.addptr %32, %31 : tensor<4x64x!tt.ptr<f32>>, tensor<4x64xi32>
    tt.store %33, %26, %18 : tensor<4x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// make_tensor_ptr load implicit permutation
module {
  tt.func public @make_tensor_ptr_implicit_perm_load_2d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %c4_i32 = arith.constant 4 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id y : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.get_program_id x : i32
    %3 = arith.muli %2, %c4_i32 : i32
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %6 = tt.splat %1 : i32 -> tensor<1x64xi32>
    %7 = arith.addi %6, %5 : tensor<1x64xi32>
    %8 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %9 = tt.expand_dims %8 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %10 = tt.splat %3 : i32 -> tensor<4x1xi32>
    %11 = arith.addi %10, %9 : tensor<4x1xi32>
    %12 = tt.splat %arg3 : i32 -> tensor<4x1xi32>
    %13 = arith.cmpi slt, %11, %12 : tensor<4x1xi32>
    %14 = tt.splat %arg2 : i32 -> tensor<1x64xi32>
    %15 = arith.cmpi slt, %7, %14 : tensor<1x64xi32>
    %16 = tt.broadcast %13 : tensor<4x1xi1> -> tensor<4x64xi1>
    %17 = tt.broadcast %15 : tensor<1x64xi1> -> tensor<4x64xi1>
    %18 = arith.andi %16, %17 : tensor<4x64xi1>
    %19 = arith.extsi %arg3 : i32 to i64
    %20 = arith.extsi %arg2 : i32 to i64
    %21 = arith.extsi %arg4 : i32 to i64
    %22 = tt.make_tensor_ptr %arg0, [%19, %20], [%c1_i64, %21], [%3, %1] {order = array<i32: 0, 1>} : <tensor<4x64xf32>>
    %23 = tt.load %22 : !tt.ptr<tensor<4x64xf32>>
    %24 = tt.splat %arg5 : i32 -> tensor<4x1xi32>
    %25 = arith.muli %11, %24 : tensor<4x1xi32>
    %26 = tt.broadcast %25 : tensor<4x1xi32> -> tensor<4x64xi32>
    %27 = tt.broadcast %7 : tensor<1x64xi32> -> tensor<4x64xi32>
    %28 = arith.addi %26, %27 : tensor<4x64xi32>
    %29 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x64x!tt.ptr<f32>>
    %30 = tt.addptr %29, %28 : tensor<4x64x!tt.ptr<f32>>, tensor<4x64xi32>
    tt.store %30, %23, %18 : tensor<4x64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// make_tensor_ptr store implicit permutation
module {
  tt.func public @make_tensor_ptr_implicit_perm_store_2d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant dense<0.000000e+00> : tensor<4x64xf32>
    %c4_i32 = arith.constant 4 : i32
    %c64_i32 = arith.constant 64 : i32
    %0 = tt.get_program_id y : i32
    %1 = arith.muli %0, %c64_i32 : i32
    %2 = tt.get_program_id x : i32
    %3 = arith.muli %2, %c4_i32 : i32
    %4 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %6 = tt.splat %1 : i32 -> tensor<1x64xi32>
    %7 = arith.addi %6, %5 : tensor<1x64xi32>
    %8 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %9 = tt.expand_dims %8 {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
    %10 = tt.splat %3 : i32 -> tensor<4x1xi32>
    %11 = arith.addi %10, %9 : tensor<4x1xi32>
    %12 = tt.splat %arg3 : i32 -> tensor<4x1xi32>
    %13 = arith.cmpi slt, %11, %12 : tensor<4x1xi32>
    %14 = tt.splat %arg2 : i32 -> tensor<1x64xi32>
    %15 = arith.cmpi slt, %7, %14 : tensor<1x64xi32>
    %16 = tt.broadcast %13 : tensor<4x1xi1> -> tensor<4x64xi1>
    %17 = tt.broadcast %15 : tensor<1x64xi1> -> tensor<4x64xi1>
    %18 = arith.andi %16, %17 : tensor<4x64xi1>
    %19 = tt.splat %arg4 : i32 -> tensor<4x1xi32>
    %20 = arith.muli %11, %19 : tensor<4x1xi32>
    %21 = tt.broadcast %20 : tensor<4x1xi32> -> tensor<4x64xi32>
    %22 = tt.broadcast %7 : tensor<1x64xi32> -> tensor<4x64xi32>
    %23 = arith.addi %21, %22 : tensor<4x64xi32>
    %24 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x64x!tt.ptr<f32>>
    %25 = tt.addptr %24, %23 : tensor<4x64x!tt.ptr<f32>>, tensor<4x64xi32>
    %26 = tt.load %25, %18, %cst : tensor<4x64x!tt.ptr<f32>>
    %27 = arith.extsi %arg3 : i32 to i64
    %28 = arith.extsi %arg2 : i32 to i64
    %29 = arith.extsi %arg5 : i32 to i64
    %30 = tt.make_tensor_ptr %arg1, [%27, %28], [%c1_i64, %29], [%3, %1] {order = array<i32: 0, 1>} : <tensor<4x64xf32>>
    tt.store %30, %26 {boundaryCheck = array<i32: 0, 1>} : !tt.ptr<tensor<4x64xf32>>
    tt.return
  }
}
