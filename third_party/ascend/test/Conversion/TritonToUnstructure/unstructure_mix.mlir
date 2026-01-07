// RUN: triton-adapter-opt --triton-to-unstructure %s | FileCheck %s 

tt.func public @indirect_mix_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<16> : tensor<1x8xi32>
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
    %9 = tt.expand_dims %2 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %10 = tt.splat %arg3 : i32 -> tensor<1x8xi32>
    %11 = arith.muli %9, %10 : tensor<1x8xi32>
    %12 = tt.expand_dims %8 {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
    %13 = arith.extsi %11 : tensor<1x8xi32> to tensor<1x8xi64>
    %14 = tt.broadcast %13 : tensor<1x8xi64> -> tensor<16x8xi64>
    %15 = tt.broadcast %12 : tensor<16x1xi64> -> tensor<16x8xi64>
    %16 = arith.addi %14, %15 : tensor<16x8xi64>
    %17 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<16x8x!tt.ptr<f32>>
    %18 = tt.addptr %17, %16 : tensor<16x8x!tt.ptr<f32>>, tensor<16x8xi64>
    %19 = tt.load %18 : tensor<16x8x!tt.ptr<f32>>
    %20 = math.exp %19 : tensor<16x8xf32>
    %21 = tt.expand_dims %4 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %22 = arith.muli %21, %cst : tensor<1x8xi32>
    %23 = tt.expand_dims %5 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %24 = tt.broadcast %22 : tensor<1x8xi32> -> tensor<16x8xi32>
    %25 = tt.broadcast %23 : tensor<16x1xi32> -> tensor<16x8xi32>
    %26 = arith.addi %24, %25 : tensor<16x8xi32>
    %27 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x8x!tt.ptr<f32>>
    %28 = tt.addptr %27, %26 : tensor<16x8x!tt.ptr<f32>>, tensor<16x8xi32>
    tt.store %28, %20 : tensor<16x8x!tt.ptr<f32>>
    tt.return
}

// CHECK-LABEL:   tt.func public @indirect_mix_kernel(
// CHECK-SAME:        %[[VAL_0:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: !tt.ptr<i64> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[VAL_3:.*]]: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_4:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant dense<16> : tensor<1x8xi32>
// CHECK:           %[[VAL_8:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_9:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_10:.*]] = arith.muli %[[VAL_9]], %[[VAL_8]] : i32
// CHECK:           %[[VAL_11:.*]] = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
// CHECK:           %[[VAL_12:.*]] = tt.splat %[[VAL_10]] : i32 -> tensor<8xi32>
// CHECK:           %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_11]] : tensor<8xi32>
// CHECK:           %[[VAL_14:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:           %[[VAL_15:.*]] = tt.splat %[[VAL_1]] : !tt.ptr<i64> -> tensor<16x!tt.ptr<i64>>
// CHECK:           %[[VAL_16:.*]] = tt.addptr %[[VAL_15]], %[[VAL_14]] : tensor<16x!tt.ptr<i64>>, tensor<16xi32>
// CHECK:           %[[VAL_17:.*]] = tt.load %[[VAL_16]] : tensor<16x!tt.ptr<i64>>
// CHECK:           %[[VAL_18:.*]] = tt.expand_dims %[[VAL_11]] {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
// CHECK:           %[[VAL_19:.*]] = tt.splat %[[VAL_3]] : i32 -> tensor<1x8xi32>
// CHECK:           %[[VAL_20:.*]] = arith.muli %[[VAL_18]], %[[VAL_19]] : tensor<1x8xi32>
// CHECK:           %[[VAL_21:.*]] = tt.expand_dims %[[VAL_17]] {axis = 1 : i32} : tensor<16xi64> -> tensor<16x1xi64>
// CHECK:           %[[VAL_22:.*]] = arith.extsi %[[VAL_20]] : tensor<1x8xi32> to tensor<1x8xi64>
// CHECK:           %[[VAL_23:.*]] = tt.broadcast %[[VAL_22]] : tensor<1x8xi64> -> tensor<16x8xi64>
// CHECK:           %[[VAL_24:.*]] = tt.broadcast %[[VAL_21]] : tensor<16x1xi64> -> tensor<16x8xi64>
// CHECK:           %[[VAL_25:.*]] = arith.addi %[[VAL_23]], %[[VAL_24]] : tensor<16x8xi64>
// CHECK:           %[[VAL_26:.*]] = tensor.empty() : tensor<16x8xf32>
// CHECK:           %[[VAL_27:.*]] = scf.for %[[VAL_28:.*]] = %[[VAL_6]] to %[[VAL_4]] step %[[VAL_5]] iter_args(%[[VAL_29:.*]] = %[[VAL_26]]) -> (tensor<16x8xf32>) {
// CHECK:             %[[VAL_30:.*]] = tensor.extract_slice %[[VAL_25]]{{\[}}%[[VAL_28]], 0] [1, 8] [1, 1] {DiscreteMemAccess} : tensor<16x8xi64> to tensor<1x8xi64>
// CHECK:             %[[VAL_31:.*]] = tt.splat %[[VAL_2]] : !tt.ptr<f32> -> tensor<1x8x!tt.ptr<f32>>
// CHECK:             %[[VAL_32:.*]] = tt.addptr %[[VAL_31]], %[[VAL_30]] : tensor<1x8x!tt.ptr<f32>>, tensor<1x8xi64>
// CHECK:             %[[VAL_33:.*]] = tt.load %[[VAL_32]] {DiscreteMemAccess} : tensor<1x8x!tt.ptr<f32>>
// CHECK:             %[[VAL_34:.*]] = tensor.insert_slice %[[VAL_33]] into %[[VAL_29]]{{\[}}%[[VAL_28]], 0] [1, 8] [1, 1] : tensor<1x8xf32> into tensor<16x8xf32>
// CHECK:             scf.yield {DiscreteMemAccess} %[[VAL_34]] : tensor<16x8xf32>
// CHECK:           } {ExtractedLoadOrStore}
// CHECK:           %[[VAL_35:.*]] = math.exp %[[VAL_27]] : tensor<16x8xf32>
// CHECK:           %[[VAL_36:.*]] = tt.expand_dims %[[VAL_13]] {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
// CHECK:           %[[VAL_37:.*]] = arith.muli %[[VAL_36]], %[[VAL_7]] : tensor<1x8xi32>
// CHECK:           %[[VAL_38:.*]] = tt.expand_dims %[[VAL_14]] {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
// CHECK:           %[[VAL_39:.*]] = tt.broadcast %[[VAL_37]] : tensor<1x8xi32> -> tensor<16x8xi32>
// CHECK:           %[[VAL_40:.*]] = tt.broadcast %[[VAL_38]] : tensor<16x1xi32> -> tensor<16x8xi32>
// CHECK:           %[[VAL_41:.*]] = arith.addi %[[VAL_39]], %[[VAL_40]] : tensor<16x8xi32>
// CHECK:           %[[VAL_42:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<f32> -> tensor<16x8x!tt.ptr<f32>>
// CHECK:           %[[VAL_43:.*]] = tt.addptr %[[VAL_42]], %[[VAL_41]] : tensor<16x8x!tt.ptr<f32>>, tensor<16x8xi32>
// CHECK:           tt.store %[[VAL_43]], %[[VAL_35]] : tensor<16x8x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }