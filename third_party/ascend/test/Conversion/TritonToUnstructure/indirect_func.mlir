// RUN: triton-adapter-opt --triton-linearize '--discrete-mask-access-conversion=compile-on-910-95=True force-simt-template=True' --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=True force-simt-template=True' %s --split-input-file | FileCheck %s

// tt.load -> tt.indirect_load
tt.func public @triton_ldst_indirect_05_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
  %cst = arith.constant dense<16> : tensor<8x1xi32>
  %c8_i32 = arith.constant 8 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c8_i32 : i32
  %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %3 = tt.splat %1 : i32 -> tensor<8xi32>
  %4 = arith.addi %3, %2 : tensor<8xi32>
  %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>>
  %7 = tt.addptr %6, %5 : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
  %8 = tt.load %7 : tensor<16x!tt.ptr<i64>>
  %9 = tt.expand_dims %2 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %10 = tt.splat %arg3 : i32 -> tensor<8x1xi32>
  %11 = arith.muli %9, %10 : tensor<8x1xi32>
  %12 = tt.expand_dims %8 {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64>
  %13 = arith.extsi %11 : tensor<8x1xi32> to tensor<8x1xi64>
  %14 = tt.broadcast %13 : tensor<8x1xi64> -> tensor<8x16xi64>
  %15 = tt.broadcast %12 : tensor<1x16xi64> -> tensor<8x16xi64>
  %16 = arith.addi %14, %15 : tensor<8x16xi64>
  %17 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
  %18 = tt.addptr %17, %16 : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi64>
  %19 = tt.load %18 : tensor<8x16x!tt.ptr<f32>>
  %20 = math.exp %19 : tensor<8x16xf32>
  %21 = tt.expand_dims %4 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %22 = arith.muli %21, %cst : tensor<8x1xi32>
  %23 = tt.expand_dims %5 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
  %24 = tt.broadcast %22 : tensor<8x1xi32> -> tensor<8x16xi32>
  %25 = tt.broadcast %23 : tensor<1x16xi32> -> tensor<8x16xi32>
  %26 = arith.addi %24, %25 : tensor<8x16xi32>
  %27 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
  %28 = tt.addptr %27, %26 : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
  tt.store %28, %20 : tensor<8x16x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL:     tt.func public @triton_ldst_indirect_05_kernel(
// CHECK-SAME:        %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<i64>, %[[VAL_2:.*]]: !tt.ptr<f32>, %[[VAL_3:.*]]: i32) attributes {noinline = false} {
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<16> : tensor<8x1xi32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_6:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_7:.*]] = arith.muli %[[VAL_6]], %[[VAL_5]] : i32
// CHECK:           %[[VAL_8:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK:           %[[VAL_9:.*]] = tt.splat %[[VAL_7]] : i32 -> tensor<8xi32>
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_8]] : tensor<8xi32>
// CHECK:           %[[VAL_11:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:           %[[VAL_12:.*]] = tt.splat %[[VAL_1:.*]] : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>>
// CHECK:           %[[VAL_13:.*]] = tt.addptr %[[VAL_12]], %[[VAL_11]] : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
// CHECK:           %[[VAL_14:.*]] = tt.load %[[VAL_13]] : tensor<16x!tt.ptr<i64>>
// CHECK:           %[[VAL_15:.*]] = tt.expand_dims %[[VAL_8]] {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
// CHECK:           %[[VAL_16:.*]] = tt.splat %[[VAL_3:.*]] : i32 -> tensor<8x1xi32>
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_15]], %[[VAL_16]] : tensor<8x1xi32>
// CHECK:           %[[VAL_18:.*]] = tt.expand_dims %[[VAL_14]] {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64>
// CHECK:           %[[VAL_19:.*]] = arith.extsi %[[VAL_17]] : tensor<8x1xi32> to tensor<8x1xi64>
// CHECK:           %[[VAL_20:.*]] = tt.broadcast %[[VAL_19]] : tensor<8x1xi64> -> tensor<8x16xi64>
// CHECK:           %[[VAL_21:.*]] = tt.broadcast %[[VAL_18]] : tensor<1x16xi64> -> tensor<8x16xi64>
// CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_20]], %[[VAL_21]] : tensor<8x16xi64>
// CHECK:           %[[VAL_23:.*]] = tt.indirect_load %[[VAL_2:.*]] : <f32>, %[[VAL_22]] : tensor<8x16xi64> -> tensor<8x16xf32>
// CHECK:           %[[VAL_24:.*]] = math.exp %[[VAL_23]] : tensor<8x16xf32>
// CHECK:           %[[VAL_25:.*]] = tt.expand_dims %[[VAL_10]] {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
// CHECK:           %[[VAL_26:.*]] = arith.muli %[[VAL_25]], %[[VAL_4]] : tensor<8x1xi32>
// CHECK:           %[[VAL_27:.*]] = tt.expand_dims %[[VAL_11]] {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
// CHECK:           %[[VAL_28:.*]] = tt.broadcast %[[VAL_26]] : tensor<8x1xi32> -> tensor<8x16xi32>
// CHECK:           %[[VAL_29:.*]] = tt.broadcast %[[VAL_27]] : tensor<1x16xi32> -> tensor<8x16xi32>
// CHECK:           %[[VAL_30:.*]] = arith.addi %[[VAL_28]], %[[VAL_29]] : tensor<8x16xi32>
// CHECK:           %[[VAL_31:.*]] = tt.splat %[[VAL_0:.*]] : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
// CHECK:           %[[VAL_32:.*]] = tt.addptr %[[VAL_31]], %[[VAL_30]] : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi32>
// CHECK:           tt.store %[[VAL_32]], %[[VAL_24]] : tensor<8x16x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }

// -----

// tt.store -> tt.indirect_store
tt.func public @triton_ldst_indirect_08_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
  %cst = arith.constant dense<32> : tensor<8x1xi32>
  %c8_i32 = arith.constant 8 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c8_i32 : i32
  %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
  %3 = tt.splat %1 : i32 -> tensor<8xi32>
  %4 = arith.addi %3, %2 : tensor<8xi32>
  %5 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
  %6 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>>
  %7 = tt.addptr %6, %5 : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
  %8 = tt.load %7 : tensor<16x!tt.ptr<i64>>
  %9 = tt.expand_dims %2 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %10 = tt.splat %arg3 : i32 -> tensor<8x1xi32>
  %11 = arith.muli %9, %10 : tensor<8x1xi32>
  %12 = tt.expand_dims %8 {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64>
  %13 = arith.extsi %11 : tensor<8x1xi32> to tensor<8x1xi64>
  %14 = tt.broadcast %13 : tensor<8x1xi64> -> tensor<8x16xi64>
  %15 = tt.broadcast %12 : tensor<1x16xi64> -> tensor<8x16xi64>
  %16 = arith.addi %14, %15 : tensor<8x16xi64>
  %17 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
  %18 = tt.addptr %17, %16 : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi64>
  %19 = tt.load %18 : tensor<8x16x!tt.ptr<f32>>
  %20 = math.exp %19 : tensor<8x16xf32>
  %21 = tt.expand_dims %4 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
  %22 = arith.muli %21, %cst : tensor<8x1xi32>
  %23 = arith.extsi %22 : tensor<8x1xi32> to tensor<8x1xi64>
  %24 = tt.broadcast %23 : tensor<8x1xi64> -> tensor<8x16xi64>
  %25 = arith.addi %24, %15 : tensor<8x16xi64>
  %26 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x16x!tt.ptr<f32>>
  %27 = tt.addptr %26, %25 : tensor<8x16x!tt.ptr<f32>>, tensor<8x16xi64>
  tt.store %27, %20 : tensor<8x16x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL:     tt.func public @triton_ldst_indirect_08_kernel(
// CHECK-SAME:       %[[VAL_0:.*]]: !tt.ptr<f32>, %[[VAL_1:.*]]: !tt.ptr<i64>, %[[VAL_2:.*]]: !tt.ptr<f32>, %[[VAL_3:.*]]: i32) attributes {noinline = false} {
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<32> : tensor<8x1xi32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_6:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_7:.*]] = arith.muli %[[VAL_6:.*]], %[[VAL_5:.*]] : i32
// CHECK:           %[[VAL_8:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK:           %[[VAL_9:.*]] = tt.splat %[[VAL_7:.*]] : i32 -> tensor<8xi32>
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_9:.*]], %[[VAL_8:.*]] : tensor<8xi32>
// CHECK:           %[[VAL_11:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:           %[[VAL_12:.*]] = tt.splat %[[VAL_1:.*]] : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>>
// CHECK:           %[[VAL_13:.*]] = tt.addptr %[[VAL_12:.*]], %[[VAL_11:.*]] : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
// CHECK:           %[[VAL_14:.*]] = tt.load %[[VAL_13:.*]] : tensor<16x!tt.ptr<i64>>
// CHECK:           %[[VAL_15:.*]] = tt.expand_dims %[[VAL_8:.*]] {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
// CHECK:           %[[VAL_16:.*]] = tt.splat %[[VAL_3:.*]] : i32 -> tensor<8x1xi32>
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_15:.*]], %[[VAL_16:.*]] : tensor<8x1xi32>
// CHECK:           %[[VAL_18:.*]] = tt.expand_dims %[[VAL_14:.*]] {axis = 0 : i32} : tensor<16xi64> -> tensor<1x16xi64>
// CHECK:           %[[VAL_19:.*]] = arith.extsi %[[VAL_17:.*]] : tensor<8x1xi32> to tensor<8x1xi64>
// CHECK:           %[[VAL_20:.*]] = tt.broadcast %[[VAL_19:.*]] : tensor<8x1xi64> -> tensor<8x16xi64>
// CHECK:           %[[VAL_21:.*]] = tt.broadcast %[[VAL_18:.*]] : tensor<1x16xi64> -> tensor<8x16xi64>
// CHECK:           %[[VAL_22:.*]] = arith.addi %[[VAL_20:.*]], %[[VAL_21:.*]] : tensor<8x16xi64>
// CHECK:           %[[VAL_23:.*]] = tt.indirect_load %[[VAL_2:.*]] : <f32>, %[[VAL_22:.*]] : tensor<8x16xi64> -> tensor<8x16xf32>
// CHECK:           %[[VAL_24:.*]] = math.exp %[[VAL_23:.*]] : tensor<8x16xf32>
// CHECK:           %[[VAL_25:.*]] = tt.expand_dims %[[VAL_10:.*]] {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
// CHECK:           %[[VAL_26:.*]] = arith.muli %[[VAL_25:.*]], %[[VAL_4:.*]] : tensor<8x1xi32>
// CHECK:           %[[VAL_27:.*]] = arith.extsi %[[VAL_26:.*]] : tensor<8x1xi32> to tensor<8x1xi64>
// CHECK:           %[[VAL_28:.*]] = tt.broadcast %[[VAL_27:.*]] : tensor<8x1xi64> -> tensor<8x16xi64>
// CHECK:           %[[VAL_29:.*]] = arith.addi %[[VAL_28:.*]], %[[VAL_21:.*]] : tensor<8x16xi64>
// CHECK:           tt.indirect_store %[[VAL_0:.*]] : <f32>, %[[VAL_29:.*]] : tensor<8x16xi64>, %[[VAL_24:.*]] : tensor<8x16xf32>
// CHECK:           tt.return
// CHECK:         }

