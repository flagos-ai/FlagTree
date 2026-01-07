// RUN: triton-adapter-opt --triton-to-unstructure --split-input-file %s | FileCheck %s

tt.func public @test_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %c3_i32 = arith.constant 3 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<128> : tensor<128xi32>
  %cst_0 = arith.constant dense<0> : tensor<128xi32>
  %cst_1 = arith.constant dense<300> : tensor<128xi32>
  %c128_i32 = arith.constant 128 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c128_i32 : i32
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %3 = tt.splat %1 : i32 -> tensor<128xi32>
  %4 = arith.addi %3, %2 : tensor<128xi32>
  %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %7 = tt.addptr %6, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %8 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %9 = tt.addptr %8, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %10 = tt.load %9 : tensor<128x!tt.ptr<i32>>
  %11 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %12 = tt.addptr %11, %10 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %13:3 = scf.for %arg4 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg5 = %4, %arg6 = %7, %arg7 = %12) -> (tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>)  : i32 {
    %14:3 = scf.for %arg8 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg9 = %arg5, %arg10 = %arg6, %arg11 = %arg7) -> (tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>)  : i32 {
      %18 = arith.cmpi slt, %arg9, %cst_1 : tensor<128xi32>
      %19 = tt.addptr %5, %arg9 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %20 = tt.load %19, %18, %cst_0 : tensor<128x!tt.ptr<i32>>
      %21 = tt.load %arg11 : tensor<128x!tt.ptr<i32>>
      %22 = arith.addi %20, %21 : tensor<128xi32>
      tt.store %arg10, %22, %18 : tensor<128x!tt.ptr<i32>>
      %23 = arith.addi %arg9, %cst : tensor<128xi32>
      %24 = tt.addptr %arg10, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %25 = tt.addptr %arg11, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      scf.yield %23, %24, %25 : tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>
    }
    %15 = arith.addi %14#0, %cst : tensor<128xi32>
    %16 = tt.addptr %14#1, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %17 = tt.addptr %14#2, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    scf.yield %15, %16, %17 : tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>
  }
  tt.return
}

// CHECK-LABEL:   tt.func public @test_kernel(
// CHECK-SAME:                                %[[VAL_0:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %[[VAL_3:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<128> : tensor<128xi64>
// CHECK:           %[[VAL_5:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_9:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_11:.*]] = arith.constant dense<128> : tensor<128xi32>
// CHECK:           %[[VAL_12:.*]] = arith.constant dense<0> : tensor<128xi32>
// CHECK:           %[[VAL_13:.*]] = arith.constant dense<300> : tensor<128xi32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 128 : i32
// CHECK:           %[[VAL_15:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_16:.*]] = arith.muli %[[VAL_15]], %[[VAL_14]] : i32
// CHECK:           %[[VAL_17:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK:           %[[VAL_18:.*]] = tt.splat %[[VAL_16]] : i32 -> tensor<128xi32>
// CHECK:           %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_17]] : tensor<128xi32>
// CHECK:           %[[VAL_20:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
// CHECK:           %[[VAL_21:.*]] = arith.extsi %[[VAL_19]] : tensor<128xi32> to tensor<128xi64>
// CHECK:           %[[VAL_22:.*]] = tt.splat %[[VAL_2]] : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
// CHECK:           %[[VAL_23:.*]] = tt.addptr %[[VAL_22]], %[[VAL_19]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
// CHECK:           %[[VAL_24:.*]] = tt.load %[[VAL_23]] : tensor<128x!tt.ptr<i32>>
// CHECK:           %[[VAL_25:.*]] = arith.extsi %[[VAL_24]] : tensor<128xi32> to tensor<128xi64>
// CHECK:           %[[VAL_26:.*]]:3 = scf.for %[[VAL_27:.*]] = %[[VAL_10]] to %[[VAL_8]] step %[[VAL_9]] iter_args(%[[VAL_28:.*]] = %[[VAL_19]], %[[VAL_29:.*]] = %[[VAL_21]], %[[VAL_30:.*]] = %[[VAL_25]]) -> (tensor<128xi32>, tensor<128xi64>, tensor<128xi64>)  : i32 {
// CHECK:             %[[VAL_31:.*]]:3 = scf.for %[[VAL_32:.*]] = %[[VAL_10]] to %[[VAL_8]] step %[[VAL_9]] iter_args(%[[VAL_33:.*]] = %[[VAL_28]], %[[VAL_34:.*]] = %[[VAL_29]], %[[VAL_35:.*]] = %[[VAL_30]]) -> (tensor<128xi32>, tensor<128xi64>, tensor<128xi64>)  : i32 {
// CHECK:               %[[VAL_36:.*]] = tt.splat %[[VAL_1]] : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
// CHECK:               %[[VAL_37:.*]] = tt.addptr %[[VAL_36]], %[[VAL_34]] : tensor<128x!tt.ptr<i32>>, tensor<128xi64>
// CHECK:               %[[VAL_38:.*]] = arith.cmpi slt, %[[VAL_33]], %[[VAL_13]] : tensor<128xi32>
// CHECK:               %[[VAL_39:.*]] = tt.addptr %[[VAL_20]], %[[VAL_33]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
// CHECK:               %[[VAL_40:.*]] = tt.load %[[VAL_39]], %[[VAL_38]], %[[VAL_12]] : tensor<128x!tt.ptr<i32>>
// CHECK:               %[[VAL_41:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:               %[[VAL_42:.*]] = scf.for %[[VAL_43:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] iter_args(%[[VAL_44:.*]] = %[[VAL_41]]) -> (tensor<128xi32>) {
// CHECK:                 %[[VAL_45:.*]] = tensor.extract %[[VAL_35]]{{\[}}%[[VAL_43]]] {DiscreteMemAccess} : tensor<128xi64>
// CHECK:                 %[[VAL_46:.*]] = tt.addptr %[[VAL_3]], %[[VAL_45]] : !tt.ptr<i32>, i64
// CHECK:                 %[[VAL_47:.*]] = tt.load %[[VAL_46]] {DiscreteMemAccess} : !tt.ptr<i32>
// CHECK:                 %[[VAL_48:.*]] = tt.splat %[[VAL_47]] : i32 -> tensor<1xi32>
// CHECK:                 %[[VAL_49:.*]] = tensor.insert_slice %[[VAL_48]] into %[[VAL_44]]{{\[}}%[[VAL_43]]] [1] [1] : tensor<1xi32> into tensor<128xi32>
// CHECK:                 scf.yield {DiscreteMemAccess} %[[VAL_49]] : tensor<128xi32>
// CHECK:               } {ExtractedLoadOrStore}
// CHECK:               %[[VAL_50:.*]] = arith.addi %[[VAL_40]], %[[VAL_42]] : tensor<128xi32>
// CHECK:               tt.store %[[VAL_37]], %[[VAL_50]], %[[VAL_38]] : tensor<128x!tt.ptr<i32>>
// CHECK:               %[[VAL_51:.*]] = arith.addi %[[VAL_33]], %[[VAL_11]] : tensor<128xi32>
// CHECK:               %[[VAL_52:.*]] = arith.addi %[[VAL_34]], %[[VAL_4]] : tensor<128xi64>
// CHECK:               %[[VAL_53:.*]] = arith.addi %[[VAL_35]], %[[VAL_4]] : tensor<128xi64>
// CHECK:               scf.yield %[[VAL_51]], %[[VAL_52]], %[[VAL_53]] : tensor<128xi32>, tensor<128xi64>, tensor<128xi64>
// CHECK:             }
// CHECK:             %[[VAL_54:.*]] = arith.addi %[[VAL_55:.*]]#0, %[[VAL_11]] : tensor<128xi32>
// CHECK:             %[[VAL_56:.*]] = arith.addi %[[VAL_55]]#1, %[[VAL_4]] : tensor<128xi64>
// CHECK:             %[[VAL_57:.*]] = arith.addi %[[VAL_55]]#2, %[[VAL_4]] : tensor<128xi64>
// CHECK:             scf.yield %[[VAL_54]], %[[VAL_56]], %[[VAL_57]] : tensor<128xi32>, tensor<128xi64>, tensor<128xi64>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }

// -----

tt.func public @test_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %c300_i32 = arith.constant 300 : i32
  %c3_i32 = arith.constant 3 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant dense<128> : tensor<128xi32>
  %cst_0 = arith.constant dense<0> : tensor<128xi32>
  %cst_1 = arith.constant dense<300> : tensor<128xi32>
  %c128_i32 = arith.constant 128 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c128_i32 : i32
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %3 = tt.splat %1 : i32 -> tensor<128xi32>
  %4 = arith.addi %3, %2 : tensor<128xi32>
  %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %6 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %7 = tt.addptr %6, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %8 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %9 = tt.addptr %8, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %10 = tt.load %9 : tensor<128x!tt.ptr<i32>>
  %11 = tt.splat %arg3 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %12 = tt.addptr %11, %10 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %13:3 = scf.while (%arg4 = %cst, %arg5 = %4, %arg6 = %7, %arg7 = %12) : (tensor<128xi32>, tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>) -> (tensor<128x!tt.ptr<i32>>, tensor<128xi32>, tensor<128x!tt.ptr<i32>>) {
    %14 = "tt.reduce"(%arg4) <{axis = 0 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %16 = arith.addi %arg8, %arg9 : i32
      tt.reduce.return %16 : i32
    }) : (tensor<128xi32>) -> i32
    %15 = arith.cmpi slt, %14, %c300_i32 : i32
    scf.condition(%15) %arg7, %arg5, %arg6 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>, tensor<128x!tt.ptr<i32>>
  } do {
  ^bb0(%arg4: tensor<128x!tt.ptr<i32>>, %arg5: tensor<128xi32>, %arg6: tensor<128x!tt.ptr<i32>>):
    %14:4 = scf.while (%arg7 = %c0_i32, %arg8 = %arg6, %arg9 = %arg4, %arg10 = %arg5) : (i32, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>, tensor<128xi32>) -> (tensor<128xi32>, i32, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>) {
      %18 = arith.cmpi slt, %arg7, %c3_i32 : i32
      scf.condition(%18) %arg10, %arg7, %arg8, %arg9 : tensor<128xi32>, i32, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>
    } do {
    ^bb0(%arg7: tensor<128xi32>, %arg8: i32, %arg9: tensor<128x!tt.ptr<i32>>, %arg10: tensor<128x!tt.ptr<i32>>):
      %18 = arith.cmpi slt, %arg7, %cst_1 : tensor<128xi32>
      %19 = tt.addptr %5, %arg7 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %20 = tt.load %19, %18, %cst_0 : tensor<128x!tt.ptr<i32>>
      %21 = tt.load %arg10 : tensor<128x!tt.ptr<i32>>
      %22 = arith.addi %20, %21 : tensor<128xi32>
      tt.store %arg9, %22, %18 : tensor<128x!tt.ptr<i32>>
      %23 = arith.addi %arg7, %cst : tensor<128xi32>
      %24 = arith.addi %arg8, %c1_i32 : i32
      %25 = tt.addptr %arg9, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %26 = tt.addptr %arg10, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      scf.yield %24, %25, %26, %23 : i32, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    }
    %15 = arith.addi %14#0, %cst : tensor<128xi32>
    %16 = tt.addptr %14#2, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    %17 = tt.addptr %14#3, %cst : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
    scf.yield %14#0, %15, %16, %17 : tensor<128xi32>, tensor<128xi32>, tensor<128x!tt.ptr<i32>>, tensor<128x!tt.ptr<i32>>
  }
  tt.return
}

// CHECK-LABEL:   tt.func public @test_kernel(
// CHECK-SAME:                                %[[VAL_0:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %[[VAL_2:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %[[VAL_3:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<128> : tensor<128xi64>
// CHECK:           %[[VAL_5:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = arith.constant 300 : i32
// CHECK:           %[[VAL_9:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_10:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_11:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_12:.*]] = arith.constant dense<128> : tensor<128xi32>
// CHECK:           %[[VAL_13:.*]] = arith.constant dense<0> : tensor<128xi32>
// CHECK:           %[[VAL_14:.*]] = arith.constant dense<300> : tensor<128xi32>
// CHECK:           %[[VAL_15:.*]] = arith.constant 128 : i32
// CHECK:           %[[VAL_16:.*]] = tt.get_program_id x : i32
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_16]], %[[VAL_15]] : i32
// CHECK:           %[[VAL_18:.*]] = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
// CHECK:           %[[VAL_19:.*]] = tt.splat %[[VAL_17]] : i32 -> tensor<128xi32>
// CHECK:           %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_18]] : tensor<128xi32>
// CHECK:           %[[VAL_21:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
// CHECK:           %[[VAL_22:.*]] = arith.extsi %[[VAL_20]] : tensor<128xi32> to tensor<128xi64>
// CHECK:           %[[VAL_23:.*]] = tt.splat %[[VAL_2]] : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
// CHECK:           %[[VAL_24:.*]] = tt.addptr %[[VAL_23]], %[[VAL_20]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
// CHECK:           %[[VAL_25:.*]] = tt.load %[[VAL_24]] : tensor<128x!tt.ptr<i32>>
// CHECK:           %[[VAL_26:.*]] = arith.extsi %[[VAL_25]] : tensor<128xi32> to tensor<128xi64>
// CHECK:           %[[VAL_27:.*]]:3 = scf.while (%[[VAL_28:.*]] = %[[VAL_12]], %[[VAL_29:.*]] = %[[VAL_20]], %[[VAL_30:.*]] = %[[VAL_22]], %[[VAL_31:.*]] = %[[VAL_26]]) : (tensor<128xi32>, tensor<128xi32>, tensor<128xi64>, tensor<128xi64>) -> (tensor<128xi64>, tensor<128xi32>, tensor<128xi64>) {
// CHECK:             %[[VAL_32:.*]] = "tt.reduce"(%[[VAL_28]]) <{axis = 0 : i32}> ({
// CHECK:             ^bb0(%[[VAL_33:.*]]: i32, %[[VAL_34:.*]]: i32):
// CHECK:               %[[VAL_35:.*]] = arith.addi %[[VAL_33]], %[[VAL_34]] : i32
// CHECK:               tt.reduce.return %[[VAL_35]] : i32
// CHECK:             }) : (tensor<128xi32>) -> i32
// CHECK:             %[[VAL_36:.*]] = arith.cmpi slt, %[[VAL_32]], %[[VAL_8]] : i32
// CHECK:             scf.condition(%[[VAL_36]]) %[[VAL_31]], %[[VAL_29]], %[[VAL_30]] : tensor<128xi64>, tensor<128xi32>, tensor<128xi64>
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_37:.*]]: tensor<128xi64>, %[[VAL_38:.*]]: tensor<128xi32>, %[[VAL_39:.*]]: tensor<128xi64>):
// CHECK:             %[[VAL_40:.*]]:4 = scf.while (%[[VAL_41:.*]] = %[[VAL_11]], %[[VAL_42:.*]] = %[[VAL_39]], %[[VAL_43:.*]] = %[[VAL_37]], %[[VAL_44:.*]] = %[[VAL_38]]) : (i32, tensor<128xi64>, tensor<128xi64>, tensor<128xi32>) -> (i32, tensor<128xi64>, tensor<128xi64>, tensor<128xi32>) {
// CHECK:               %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_41]], %[[VAL_9]] : i32
// CHECK:               scf.condition(%[[VAL_45]]) %[[VAL_41]], %[[VAL_42]], %[[VAL_43]], %[[VAL_44]] : i32, tensor<128xi64>, tensor<128xi64>, tensor<128xi32>
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_46:.*]]: i32, %[[VAL_47:.*]]: tensor<128xi64>, %[[VAL_48:.*]]: tensor<128xi64>, %[[VAL_49:.*]]: tensor<128xi32>):
// CHECK:               %[[VAL_50:.*]] = tt.splat %[[VAL_1]] : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
// CHECK:               %[[VAL_51:.*]] = tt.addptr %[[VAL_50]], %[[VAL_47]] : tensor<128x!tt.ptr<i32>>, tensor<128xi64>
// CHECK:               %[[VAL_52:.*]] = arith.cmpi slt, %[[VAL_49]], %[[VAL_14]] : tensor<128xi32>
// CHECK:               %[[VAL_53:.*]] = tt.addptr %[[VAL_21]], %[[VAL_49]] : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
// CHECK:               %[[VAL_54:.*]] = tt.load %[[VAL_53]], %[[VAL_52]], %[[VAL_13]] : tensor<128x!tt.ptr<i32>>
// CHECK:               %[[VAL_55:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:               %[[VAL_56:.*]] = scf.for %[[VAL_57:.*]] = %[[VAL_7]] to %[[VAL_5]] step %[[VAL_6]] iter_args(%[[VAL_58:.*]] = %[[VAL_55]]) -> (tensor<128xi32>) {
// CHECK:                 %[[VAL_59:.*]] = tensor.extract %[[VAL_48]]{{\[}}%[[VAL_57]]] {DiscreteMemAccess} : tensor<128xi64>
// CHECK:                 %[[VAL_60:.*]] = tt.addptr %[[VAL_3]], %[[VAL_59]] : !tt.ptr<i32>, i64
// CHECK:                 %[[VAL_61:.*]] = tt.load %[[VAL_60]] {DiscreteMemAccess} : !tt.ptr<i32>
// CHECK:                 %[[VAL_62:.*]] = tt.splat %[[VAL_61]] : i32 -> tensor<1xi32>
// CHECK:                 %[[VAL_63:.*]] = tensor.insert_slice %[[VAL_62]] into %[[VAL_58]]{{\[}}%[[VAL_57]]] [1] [1] : tensor<1xi32> into tensor<128xi32>
// CHECK:                 scf.yield {DiscreteMemAccess} %[[VAL_63]] : tensor<128xi32>
// CHECK:               } {ExtractedLoadOrStore}
// CHECK:               %[[VAL_64:.*]] = arith.addi %[[VAL_54]], %[[VAL_56]] : tensor<128xi32>
// CHECK:               tt.store %[[VAL_51]], %[[VAL_64]], %[[VAL_52]] : tensor<128x!tt.ptr<i32>>
// CHECK:               %[[VAL_65:.*]] = arith.addi %[[VAL_49]], %[[VAL_12]] : tensor<128xi32>
// CHECK:               %[[VAL_66:.*]] = arith.addi %[[VAL_46]], %[[VAL_10]] : i32
// CHECK:               %[[VAL_67:.*]] = arith.addi %[[VAL_47]], %[[VAL_4]] : tensor<128xi64>
// CHECK:               %[[VAL_68:.*]] = arith.addi %[[VAL_48]], %[[VAL_4]] : tensor<128xi64>
// CHECK:               scf.yield %[[VAL_66]], %[[VAL_67]], %[[VAL_68]], %[[VAL_65]] : i32, tensor<128xi64>, tensor<128xi64>, tensor<128xi32>
// CHECK:             }
// CHECK:             %[[VAL_69:.*]] = arith.addi %[[VAL_70:.*]]#3, %[[VAL_12]] : tensor<128xi32>
// CHECK:             %[[VAL_71:.*]] = arith.addi %[[VAL_70]]#1, %[[VAL_4]] : tensor<128xi64>
// CHECK:             %[[VAL_72:.*]] = arith.addi %[[VAL_70]]#2, %[[VAL_4]] : tensor<128xi64>
// CHECK:             scf.yield %[[VAL_70]]#3, %[[VAL_69]], %[[VAL_71]], %[[VAL_72]] : tensor<128xi32>, tensor<128xi32>, tensor<128xi64>, tensor<128xi64>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }