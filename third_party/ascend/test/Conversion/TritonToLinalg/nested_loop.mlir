// RUN: triton-adapter-opt --triton-to-linalg --split-input-file %s | FileCheck %s

tt.func public @test_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<128> : tensor<128xi64>
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant dense<0> : tensor<128xi64>
  %c3_i32 = arith.constant 3 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst_1 = arith.constant dense<128> : tensor<128xi32>
  %cst_2 = arith.constant dense<0> : tensor<128xi32>
  %cst_3 = arith.constant dense<300> : tensor<128xi32>
  %c128_i32 = arith.constant 128 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c128_i32 : i32
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %3 = tt.splat %1 : i32 -> tensor<128xi32>
  %4 = arith.addi %3, %2 : tensor<128xi32>
  %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %6 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %7 = tt.addptr %6, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %8 = tt.load %7 : tensor<128x!tt.ptr<i32>>
  %9 = arith.extsi %8 : tensor<128xi32> to tensor<128xi64>
  %10:3 = scf.for %arg4 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg5 = %4, %arg6 = %cst_0, %arg7 = %9) -> (tensor<128xi32>, tensor<128xi64>, tensor<128xi64>)  : i32 {
    %11 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
    %12 = tt.addptr %11, %arg6 : tensor<128x!tt.ptr<i32>>, tensor<128xi64>
    %13:3 = scf.for %arg8 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg9 = %arg5, %arg10 = %arg6, %arg11 = %arg7) -> (tensor<128xi32>, tensor<128xi64>, tensor<128xi64>)  : i32 {
      %19 = tt.addptr %12, %arg10 : tensor<128x!tt.ptr<i32>>, tensor<128xi64>
      %20 = arith.cmpi slt, %arg9, %cst_3 : tensor<128xi32>
      %21 = tt.addptr %5, %arg9 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %22 = tt.load %21 : tensor<128x!tt.ptr<i32>>
      %23 = arith.select %20, %22, %cst_2 : tensor<128xi1>, tensor<128xi32>
      %24 = tensor.empty() : tensor<128xi32>
      %25 = scf.for %arg12 = %c0 to %c128 step %c1 iter_args(%arg13 = %24) -> (tensor<128xi32>) {
        %extracted = tensor.extract %arg7[%arg12] {DiscreteMemAccess} : tensor<128xi64>
        %extracted_4 = tensor.extract %arg11[%arg12] {DiscreteMemAccess} : tensor<128xi64>
        %32 = arith.addi %extracted, %extracted_4 : i64
        %33 = tt.addptr %arg3, %32 : !tt.ptr<i32>, i64
        %34 = tt.load %33 {DiscreteMemAccess} : !tt.ptr<i32>
        %35 = tt.splat %34 : i32 -> tensor<1xi32>
        %inserted_slice = tensor.insert_slice %35 into %arg13[%arg12] [1] [1] : tensor<1xi32> into tensor<128xi32>
        scf.yield {DiscreteMemAccess} %inserted_slice : tensor<128xi32>
      } {ExtractedLoadOrStore}
      %26 = arith.addi %23, %25 : tensor<128xi32>
      %27 = tt.load %19 : tensor<128x!tt.ptr<i32>>
      %28 = arith.select %20, %26, %27 : tensor<128xi1>, tensor<128xi32>
      tt.store %19, %28 {DiscreteMask} : tensor<128x!tt.ptr<i32>>
      %29 = arith.addi %arg9, %cst_1 : tensor<128xi32>
      %30 = arith.addi %arg10, %cst : tensor<128xi64>
      %31 = arith.addi %arg11, %cst : tensor<128xi64>
      scf.yield %29, %30, %31 : tensor<128xi32>, tensor<128xi64>, tensor<128xi64>
    }
    %14 = arith.addi %arg7, %13#2 : tensor<128xi64>
    %15 = arith.addi %arg6, %13#1 : tensor<128xi64>
    %16 = arith.addi %13#0, %cst_1 : tensor<128xi32>
    %17 = arith.addi %15, %cst : tensor<128xi64>
    %18 = arith.addi %14, %cst : tensor<128xi64>
    scf.yield %16, %17, %18 : tensor<128xi32>, tensor<128xi64>, tensor<128xi64>
  }
  tt.return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @test_kernel(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<?xi8>, %[[VAL_1:.*]]: memref<?xi8>, %[[VAL_2:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %[[VAL_3:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 2 : i32}, %[[VAL_4:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %[[VAL_5:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:           %[[VAL_12:.*]] = arith.constant 128 : i32
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_15:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_16:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_17:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_18:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_19:.*]] = arith.constant 300 : index
// CHECK:           %[[VAL_20:.*]] = arith.constant 128 : i64
// CHECK:           %[[VAL_21:.*]] = tensor.empty() : tensor<128xi64>
// CHECK:           %[[VAL_22:.*]] = linalg.fill ins(%[[VAL_20]] : i64) outs(%[[VAL_21]] : tensor<128xi64>) -> tensor<128xi64>
// CHECK:           %[[VAL_23:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[VAL_24:.*]] = linalg.fill ins(%[[VAL_13]] : i32) outs(%[[VAL_23]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK:           %[[VAL_25:.*]] = arith.muli %[[VAL_9]], %[[VAL_12]] : i32
// CHECK:           %[[VAL_26:.*]] = arith.index_cast %[[VAL_25]] : i32 to index
// CHECK:           %[[VAL_27:.*]] = memref.reinterpret_cast %[[VAL_4]] to offset: {{\[}}%[[VAL_26]]], sizes: [128], strides: [1] : memref<?xi32> to memref<128xi32, strided<[1], offset: ?>>
// CHECK:           %[[VAL_28:.*]] = memref.alloc() : memref<128xi32>
// CHECK:           memref.copy %[[VAL_27]], %[[VAL_28]] : memref<128xi32, strided<[1], offset: ?>> to memref<128xi32>
// CHECK:           %[[VAL_29:.*]] = bufferization.to_tensor %[[VAL_28]] restrict writable : memref<128xi32>
// CHECK:           %[[VAL_30:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_29]] : tensor<128xi32>) outs(%[[VAL_21]] : tensor<128xi64>) {
// CHECK:           ^bb0(%[[VAL_31:.*]]: i32, %[[VAL_32:.*]]: i64):
// CHECK:             %[[VAL_33:.*]] = arith.extsi %[[VAL_31]] : i32 to i64
// CHECK:             linalg.yield %[[VAL_33]] : i64
// CHECK:           } -> tensor<128xi64>
// CHECK:           %[[VAL_34:.*]]:4 = scf.for %[[VAL_35:.*]] = %[[VAL_13]] to %[[VAL_15]] step %[[VAL_14]] iter_args(%[[VAL_36:.*]] = %[[VAL_30]], %[[VAL_37:.*]] = %[[VAL_26]], %[[VAL_38:.*]] = %[[VAL_16]], %[[VAL_39:.*]] = %[[VAL_16]]) -> (tensor<128xi64>, index, index, index)  : i32 {
// CHECK:             %[[VAL_40:.*]]:3 = scf.for %[[VAL_41:.*]] = %[[VAL_13]] to %[[VAL_15]] step %[[VAL_14]] iter_args(%[[VAL_42:.*]] = %[[VAL_36]], %[[VAL_43:.*]] = %[[VAL_37]], %[[VAL_44:.*]] = %[[VAL_38]]) -> (tensor<128xi64>, index, index)  : i32 {
// CHECK:               %[[VAL_45:.*]] = arith.addi %[[VAL_38]], %[[VAL_44]] : index
// CHECK:               %[[VAL_46:.*]] = arith.addi %[[VAL_39]], %[[VAL_39]] : index
// CHECK:               %[[VAL_47:.*]] = memref.reinterpret_cast %[[VAL_3]] to offset: {{\[}}%[[VAL_45]]], sizes: [128], strides: {{\[}}%[[VAL_46]]] : memref<?xi32> to memref<128xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_48:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_43]]], sizes: [128], strides: {{\[}}%[[VAL_17]]] : memref<?xi32> to memref<128xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_49:.*]] = memref.alloc() : memref<128xi32>
// CHECK:               memref.copy %[[VAL_48]], %[[VAL_49]] : memref<128xi32, strided<[?], offset: ?>> to memref<128xi32>
// CHECK:               %[[VAL_50:.*]] = bufferization.to_tensor %[[VAL_49]] restrict writable : memref<128xi32>
// CHECK:               %[[VAL_51:.*]] = arith.addi %[[VAL_26]], %[[VAL_18]] : index
// CHECK:               %[[VAL_52:.*]] = arith.maxsi %[[VAL_26]], %[[VAL_19]] : index
// CHECK:               %[[VAL_53:.*]] = arith.minsi %[[VAL_51]], %[[VAL_52]] : index
// CHECK:               %[[VAL_54:.*]] = arith.subi %[[VAL_53]], %[[VAL_26]] : index
// CHECK:               %[[VAL_55:.*]] = tensor.extract_slice %[[VAL_50]][0] {{\[}}%[[VAL_54]]] [1] : tensor<128xi32> to tensor<?xi32>
// CHECK:               %[[VAL_56:.*]] = tensor.insert_slice %[[VAL_55]] into %[[VAL_24]][0] {{\[}}%[[VAL_54]]] [1] : tensor<?xi32> into tensor<128xi32>
// CHECK:               %[[VAL_57:.*]] = scf.for %[[VAL_58:.*]] = %[[VAL_16]] to %[[VAL_18]] step %[[VAL_17]] iter_args(%[[VAL_59:.*]] = %[[VAL_23]]) -> (tensor<128xi32>) {
// CHECK:                 %[[VAL_60:.*]] = tensor.extract %[[VAL_36]]{{\[}}%[[VAL_58]]] {DiscreteMemAccess} : tensor<128xi64>
// CHECK:                 %[[VAL_61:.*]] = tensor.extract %[[VAL_42]]{{\[}}%[[VAL_58]]] {DiscreteMemAccess} : tensor<128xi64>
// CHECK:                 %[[VAL_62:.*]] = arith.addi %[[VAL_60]], %[[VAL_61]] : i64
// CHECK:                 %[[VAL_63:.*]] = arith.index_cast %[[VAL_62]] : i64 to index
// CHECK:                 %[[VAL_64:.*]] = memref.reinterpret_cast %[[VAL_5]] to offset: {{\[}}%[[VAL_63]]], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:                 %[[VAL_65:.*]] = memref.load %[[VAL_64]]{{\[}}%[[VAL_16]]] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:                 %[[VAL_66:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:                 %[[VAL_67:.*]] = linalg.fill ins(%[[VAL_65]] : i32) outs(%[[VAL_66]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:                 %[[VAL_68:.*]] = tensor.insert_slice %[[VAL_67]] into %[[VAL_59]]{{\[}}%[[VAL_58]]] [1] [1] : tensor<1xi32> into tensor<128xi32>
// CHECK:                 scf.yield {DiscreteMemAccess} %[[VAL_68]] : tensor<128xi32>
// CHECK:               } {ExtractedLoadOrStore}
// CHECK:               %[[VAL_69:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_56]], %[[VAL_57]] : tensor<128xi32>, tensor<128xi32>) outs(%[[VAL_56]] : tensor<128xi32>) {
// CHECK:               ^bb0(%[[VAL_70:.*]]: i32, %[[VAL_71:.*]]: i32, %[[VAL_72:.*]]: i32):
// CHECK:                 %[[VAL_73:.*]] = arith.addi %[[VAL_70]], %[[VAL_71]] : i32
// CHECK:                 linalg.yield %[[VAL_73]] : i32
// CHECK:               } -> tensor<128xi32>
// CHECK:               %[[VAL_74:.*]] = memref.alloc() : memref<128xi32>
// CHECK:               memref.copy %[[VAL_47]], %[[VAL_74]] : memref<128xi32, strided<[?], offset: ?>> to memref<128xi32>
// CHECK:               %[[VAL_75:.*]] = bufferization.to_tensor %[[VAL_74]] restrict writable : memref<128xi32>
// CHECK:               %[[VAL_76:.*]] = tensor.extract_slice %[[VAL_69]][0] {{\[}}%[[VAL_54]]] [1] : tensor<128xi32> to tensor<?xi32>
// CHECK:               %[[VAL_77:.*]] = tensor.insert_slice %[[VAL_76]] into %[[VAL_75]][0] {{\[}}%[[VAL_54]]] [1] : tensor<?xi32> into tensor<128xi32>
// CHECK:               bufferization.materialize_in_destination %[[VAL_77]] in writable %[[VAL_47]] : (tensor<128xi32>, memref<128xi32, strided<[?], offset: ?>>) -> ()
// CHECK:               %[[VAL_78:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_42]], %[[VAL_22]] : tensor<128xi64>, tensor<128xi64>) outs(%[[VAL_42]] : tensor<128xi64>) {
// CHECK:               ^bb0(%[[VAL_79:.*]]: i64, %[[VAL_80:.*]]: i64, %[[VAL_81:.*]]: i64):
// CHECK:                 %[[VAL_82:.*]] = arith.addi %[[VAL_79]], %[[VAL_80]] : i64
// CHECK:                 linalg.yield %[[VAL_82]] : i64
// CHECK:               } -> tensor<128xi64>
// CHECK:               %[[VAL_83:.*]] = arith.addi %[[VAL_43]], %[[VAL_18]] : index
// CHECK:               %[[VAL_84:.*]] = arith.addi %[[VAL_44]], %[[VAL_18]] : index
// CHECK:               scf.yield %[[VAL_78]], %[[VAL_83]], %[[VAL_84]] : tensor<128xi64>, index, index
// CHECK:             }
// CHECK:             %[[VAL_85:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_36]], %[[VAL_86:.*]]#0 : tensor<128xi64>, tensor<128xi64>) outs(%[[VAL_36]] : tensor<128xi64>) {
// CHECK:             ^bb0(%[[VAL_87:.*]]: i64, %[[VAL_88:.*]]: i64, %[[VAL_89:.*]]: i64):
// CHECK:               %[[VAL_90:.*]] = arith.addi %[[VAL_87]], %[[VAL_88]] : i64
// CHECK:               linalg.yield %[[VAL_90]] : i64
// CHECK:             } -> tensor<128xi64>
// CHECK:             %[[VAL_91:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_85]], %[[VAL_22]] : tensor<128xi64>, tensor<128xi64>) outs(%[[VAL_85]] : tensor<128xi64>) {
// CHECK:             ^bb0(%[[VAL_92:.*]]: i64, %[[VAL_93:.*]]: i64, %[[VAL_94:.*]]: i64):
// CHECK:               %[[VAL_95:.*]] = arith.addi %[[VAL_92]], %[[VAL_93]] : i64
// CHECK:               linalg.yield %[[VAL_95]] : i64
// CHECK:             } -> tensor<128xi64>
// CHECK:             %[[VAL_96:.*]] = arith.addi %[[VAL_97:.*]]#1, %[[VAL_18]] : index
// CHECK:             %[[VAL_98:.*]] = arith.addi %[[VAL_38]], %[[VAL_97]]#2 : index
// CHECK:             %[[VAL_99:.*]] = arith.addi %[[VAL_39]], %[[VAL_39]] : index
// CHECK:             %[[VAL_100:.*]] = arith.addi %[[VAL_98]], %[[VAL_18]] : index
// CHECK:             scf.yield %[[VAL_91]], %[[VAL_96]], %[[VAL_100]], %[[VAL_99]] : tensor<128xi64>, index, index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }

// -----

tt.func public @test_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
  %cst = arith.constant dense<128> : tensor<128xi64>
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c300_i32 = arith.constant 300 : i32
  %c3_i32 = arith.constant 3 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0_i32 = arith.constant 0 : i32
  %cst_0 = arith.constant dense<128> : tensor<128xi32>
  %cst_1 = arith.constant dense<0> : tensor<128xi32>
  %cst_2 = arith.constant dense<300> : tensor<128xi32>
  %c128_i32 = arith.constant 128 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c128_i32 : i32
  %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  %3 = tt.splat %1 : i32 -> tensor<128xi32>
  %4 = arith.addi %3, %2 : tensor<128xi32>
  %5 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %6 = arith.extsi %4 : tensor<128xi32> to tensor<128xi64>
  %7 = tt.splat %arg2 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
  %8 = tt.addptr %7, %4 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
  %9 = tt.load %8 : tensor<128x!tt.ptr<i32>>
  %10 = arith.extsi %9 : tensor<128xi32> to tensor<128xi64>
  %11:3 = scf.while (%arg4 = %cst_0, %arg5 = %4, %arg6 = %6, %arg7 = %10) : (tensor<128xi32>, tensor<128xi32>, tensor<128xi64>, tensor<128xi64>) -> (tensor<128xi64>, tensor<128xi32>, tensor<128xi64>) {
    %12 = "tt.reduce"(%arg4) <{axis = 0 : i32}> ({
    ^bb0(%arg8: i32, %arg9: i32):
      %14 = arith.addi %arg8, %arg9 : i32
      tt.reduce.return %14 : i32
    }) : (tensor<128xi32>) -> i32
    %13 = arith.cmpi slt, %12, %c300_i32 : i32
    scf.condition(%13) %arg7, %arg5, %arg6 : tensor<128xi64>, tensor<128xi32>, tensor<128xi64>
  } do {
  ^bb0(%arg4: tensor<128xi64>, %arg5: tensor<128xi32>, %arg6: tensor<128xi64>):
    %12:4 = scf.while (%arg7 = %c0_i32, %arg8 = %arg6, %arg9 = %arg4, %arg10 = %arg5) : (i32, tensor<128xi64>, tensor<128xi64>, tensor<128xi32>) -> (tensor<128xi32>, i32, tensor<128xi64>, tensor<128xi64>) {
      %16 = arith.cmpi slt, %arg7, %c3_i32 : i32
      scf.condition(%16) %arg10, %arg7, %arg8, %arg9 : tensor<128xi32>, i32, tensor<128xi64>, tensor<128xi64>
    } do {
    ^bb0(%arg7: tensor<128xi32>, %arg8: i32, %arg9: tensor<128xi64>, %arg10: tensor<128xi64>):
      %16 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>>
      %17 = tt.addptr %16, %arg9 : tensor<128x!tt.ptr<i32>>, tensor<128xi64>
      %18 = arith.cmpi slt, %arg7, %cst_2 : tensor<128xi32>
      %19 = tt.addptr %5, %arg7 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
      %20 = tt.load %19, %18, %cst_1 : tensor<128x!tt.ptr<i32>>
      %21 = tensor.empty() : tensor<128xi32>
      %22 = scf.for %arg11 = %c0 to %c128 step %c1 iter_args(%arg12 = %21) -> (tensor<128xi32>) {
        %extracted = tensor.extract %arg10[%arg11] {DiscreteMemAccess} : tensor<128xi64>
        %28 = tt.addptr %arg3, %extracted : !tt.ptr<i32>, i64
        %29 = tt.load %28 {DiscreteMemAccess} : !tt.ptr<i32>
        %30 = tt.splat %29 : i32 -> tensor<1xi32>
        %inserted_slice = tensor.insert_slice %30 into %arg12[%arg11] [1] [1] : tensor<1xi32> into tensor<128xi32>
        scf.yield {DiscreteMemAccess} %inserted_slice : tensor<128xi32>
      } {ExtractedLoadOrStore}
      %23 = arith.addi %20, %22 : tensor<128xi32>
      tt.store %17, %23, %18 : tensor<128x!tt.ptr<i32>>
      %24 = arith.addi %arg7, %cst_0 : tensor<128xi32>
      %25 = arith.addi %arg8, %c1_i32 : i32
      %26 = arith.addi %6, %cst : tensor<128xi64>
      %27 = arith.addi %10, %cst : tensor<128xi64>
      scf.yield %25, %26, %27, %24 : i32, tensor<128xi64>, tensor<128xi64>, tensor<128xi32>
    }
    %13 = arith.addi %12#0, %cst_0 : tensor<128xi32>
    %14 = arith.addi %12#2, %cst : tensor<128xi64>
    %15 = arith.addi %12#3, %cst : tensor<128xi64>
    scf.yield %12#0, %13, %14, %15 : tensor<128xi32>, tensor<128xi32>, tensor<128xi64>, tensor<128xi64>
  }
  tt.return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:   func.func @test_kernel(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<?xi8>, %[[VAL_1:.*]]: memref<?xi8>, %[[VAL_2:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %[[VAL_3:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %[[VAL_4:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %[[VAL_5:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:           %[[VAL_12:.*]] = arith.constant 300 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 128 : i32
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_15:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_16:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_17:.*]] = arith.constant 300 : i32
// CHECK:           %[[VAL_18:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_20:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_21:.*]] = arith.constant 128 : i64
// CHECK:           %[[VAL_22:.*]] = tensor.empty() : tensor<128xi64>
// CHECK:           %[[VAL_23:.*]] = linalg.fill ins(%[[VAL_21]] : i64) outs(%[[VAL_22]] : tensor<128xi64>) -> tensor<128xi64>
// CHECK:           %[[VAL_24:.*]] = tensor.empty() : tensor<128xi32>
// CHECK:           %[[VAL_25:.*]] = linalg.fill ins(%[[VAL_13]] : i32) outs(%[[VAL_24]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK:           %[[VAL_26:.*]] = arith.muli %[[VAL_9]], %[[VAL_13]] : i32
// CHECK:           %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[VAL_24]] : tensor<128xi32>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: i32):
// CHECK:             %[[VAL_29:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_30:.*]] = arith.index_cast %[[VAL_29]] : index to i32
// CHECK:             linalg.yield %[[VAL_30]] : i32
// CHECK:           } -> tensor<128xi32>
// CHECK:           %[[VAL_31:.*]] = arith.index_cast %[[VAL_26]] : i32 to index
// CHECK:           %[[VAL_32:.*]] = memref.reinterpret_cast %[[VAL_4]] to offset: {{\[}}%[[VAL_31]]], sizes: [128], strides: [1] : memref<?xi32> to memref<128xi32, strided<[1], offset: ?>>
// CHECK:           %[[VAL_33:.*]] = memref.alloc() : memref<128xi32>
// CHECK:           memref.copy %[[VAL_32]], %[[VAL_33]] : memref<128xi32, strided<[1], offset: ?>> to memref<128xi32>
// CHECK:           %[[VAL_34:.*]] = bufferization.to_tensor %[[VAL_33]] restrict writable : memref<128xi32>
// CHECK:           %[[VAL_35:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_34]] : tensor<128xi32>) outs(%[[VAL_22]] : tensor<128xi64>) {
// CHECK:           ^bb0(%[[VAL_36:.*]]: i32, %[[VAL_37:.*]]: i64):
// CHECK:             %[[VAL_38:.*]] = arith.extsi %[[VAL_36]] : i32 to i64
// CHECK:             linalg.yield %[[VAL_38]] : i64
// CHECK:           } -> tensor<128xi64>
// CHECK:           %[[VAL_39:.*]]:4 = scf.while (%[[VAL_40:.*]] = %[[VAL_25]], %[[VAL_41:.*]] = %[[VAL_35]], %[[VAL_42:.*]] = %[[VAL_31]], %[[VAL_43:.*]] = %[[VAL_31]], %[[VAL_44:.*]] = %[[VAL_19]]) : (tensor<128xi32>, tensor<128xi64>, index, index, index) -> (tensor<128xi64>, index, index, index) {
// CHECK:             %[[VAL_45:.*]] = bufferization.alloc_tensor() : tensor<i32>
// CHECK:             %[[VAL_46:.*]] = linalg.fill ins(%[[VAL_14]] : i32) outs(%[[VAL_45]] : tensor<i32>) -> tensor<i32>
// CHECK:             %[[VAL_47:.*]] = linalg.reduce ins(%[[VAL_40]] : tensor<128xi32>) outs(%[[VAL_46]] : tensor<i32>) dimensions = [0]
// CHECK:               (%[[VAL_48:.*]]: i32, %[[VAL_49:.*]]: i32) {
// CHECK:                 %[[VAL_50:.*]] = arith.addi %[[VAL_48]], %[[VAL_49]] : i32
// CHECK:                 linalg.yield %[[VAL_50]] : i32
// CHECK:               }
// CHECK:             %[[VAL_51:.*]] = tensor.extract %[[VAL_47]][] : tensor<i32>
// CHECK:             %[[VAL_52:.*]] = arith.cmpi slt, %[[VAL_51]], %[[VAL_17]] : i32
// CHECK:             scf.condition(%[[VAL_52]]) %[[VAL_41]], %[[VAL_42]], %[[VAL_43]], %[[VAL_44]] : tensor<128xi64>, index, index, index
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_53:.*]]: tensor<128xi64>, %[[VAL_54:.*]]: index, %[[VAL_55:.*]]: index, %[[VAL_56:.*]]: index):
// CHECK:             %[[VAL_57:.*]]:5 = scf.while (%[[VAL_58:.*]] = %[[VAL_14]], %[[VAL_59:.*]] = %[[VAL_53]], %[[VAL_60:.*]] = %[[VAL_55]], %[[VAL_61:.*]] = %[[VAL_56]], %[[VAL_62:.*]] = %[[VAL_54]]) : (i32, tensor<128xi64>, index, index, index) -> (i32, tensor<128xi64>, index, index, index) {
// CHECK:               %[[VAL_63:.*]] = arith.cmpi slt, %[[VAL_58]], %[[VAL_16]] : i32
// CHECK:               scf.condition(%[[VAL_63]]) %[[VAL_58]], %[[VAL_59]], %[[VAL_60]], %[[VAL_61]], %[[VAL_62]] : i32, tensor<128xi64>, index, index, index
// CHECK:             } do {
// CHECK:             ^bb0(%[[VAL_64:.*]]: i32, %[[VAL_65:.*]]: tensor<128xi64>, %[[VAL_66:.*]]: index, %[[VAL_67:.*]]: index, %[[VAL_68:.*]]: index):
// CHECK:               %[[VAL_69:.*]] = memref.reinterpret_cast %[[VAL_3]] to offset: {{\[}}%[[VAL_66]]], sizes: [128], strides: {{\[}}%[[VAL_67]]] : memref<?xi32> to memref<128xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_70:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_68]]], sizes: [128], strides: {{\[}}%[[VAL_19]]] : memref<?xi32> to memref<128xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_71:.*]] = memref.alloc() : memref<128xi32>
// CHECK:               %[[VAL_72:.*]] = arith.addi %[[VAL_68]], %[[VAL_20]] : index
// CHECK:               %[[VAL_73:.*]] = arith.maxsi %[[VAL_68]], %[[VAL_12]] : index
// CHECK:               %[[VAL_74:.*]] = arith.minsi %[[VAL_72]], %[[VAL_73]] : index
// CHECK:               %[[VAL_75:.*]] = arith.subi %[[VAL_74]], %[[VAL_68]] : index
// CHECK:               %[[VAL_76:.*]] = arith.cmpi slt, %[[VAL_75]], %[[VAL_20]] : index
// CHECK:               scf.if %[[VAL_76]] {
// CHECK:                 linalg.fill ins(%[[VAL_14]] : i32) outs(%[[VAL_71]] : memref<128xi32>)
// CHECK:               }
// CHECK:               %[[VAL_77:.*]] = memref.subview %[[VAL_70]][0] {{\[}}%[[VAL_75]]] [1] : memref<128xi32, strided<[?], offset: ?>> to memref<?xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_78:.*]] = memref.subview %[[VAL_71]][0] {{\[}}%[[VAL_75]]] [1] : memref<128xi32> to memref<?xi32, strided<[1]>>
// CHECK:               memref.copy %[[VAL_77]], %[[VAL_78]] : memref<?xi32, strided<[?], offset: ?>> to memref<?xi32, strided<[1]>>
// CHECK:               %[[VAL_79:.*]] = bufferization.to_tensor %[[VAL_71]] restrict writable : memref<128xi32>
// CHECK:               %[[VAL_80:.*]] = scf.for %[[VAL_81:.*]] = %[[VAL_18]] to %[[VAL_20]] step %[[VAL_19]] iter_args(%[[VAL_82:.*]] = %[[VAL_24]]) -> (tensor<128xi32>) {
// CHECK:                 %[[VAL_83:.*]] = tensor.extract %[[VAL_65]]{{\[}}%[[VAL_81]]] {DiscreteMemAccess} : tensor<128xi64>
// CHECK:                 %[[VAL_84:.*]] = arith.index_cast %[[VAL_83]] : i64 to index
// CHECK:                 %[[VAL_85:.*]] = memref.reinterpret_cast %[[VAL_5]] to offset: {{\[}}%[[VAL_84]]], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:                 %[[VAL_86:.*]] = memref.load %[[VAL_85]]{{\[}}%[[VAL_18]]] : memref<1xi32, strided<[1], offset: ?>>
// CHECK:                 %[[VAL_87:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:                 %[[VAL_88:.*]] = linalg.fill ins(%[[VAL_86]] : i32) outs(%[[VAL_87]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:                 %[[VAL_89:.*]] = tensor.insert_slice %[[VAL_88]] into %[[VAL_82]]{{\[}}%[[VAL_81]]] [1] [1] : tensor<1xi32> into tensor<128xi32>
// CHECK:                 scf.yield {DiscreteMemAccess} %[[VAL_89]] : tensor<128xi32>
// CHECK:               } {ExtractedLoadOrStore}
// CHECK:               %[[VAL_90:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_79]], %[[VAL_80]] : tensor<128xi32>, tensor<128xi32>) outs(%[[VAL_79]] : tensor<128xi32>) {
// CHECK:               ^bb0(%[[VAL_91:.*]]: i32, %[[VAL_92:.*]]: i32, %[[VAL_93:.*]]: i32):
// CHECK:                 %[[VAL_94:.*]] = arith.addi %[[VAL_91]], %[[VAL_92]] : i32
// CHECK:                 linalg.yield %[[VAL_94]] : i32
// CHECK:               } -> tensor<128xi32>
// CHECK:               %[[VAL_95:.*]] = tensor.extract_slice %[[VAL_90]][0] {{\[}}%[[VAL_75]]] [1] : tensor<128xi32> to tensor<?xi32>
// CHECK:               %[[VAL_96:.*]] = memref.subview %[[VAL_69]][0] {{\[}}%[[VAL_75]]] [1] : memref<128xi32, strided<[?], offset: ?>> to memref<?xi32, strided<[?], offset: ?>>
// CHECK:               bufferization.materialize_in_destination %[[VAL_95]] in writable %[[VAL_96]] : (tensor<?xi32>, memref<?xi32, strided<[?], offset: ?>>) -> ()
// CHECK:               %[[VAL_97:.*]] = arith.addi %[[VAL_64]], %[[VAL_15]] : i32
// CHECK:               %[[VAL_98:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_35]], %[[VAL_23]] : tensor<128xi64>, tensor<128xi64>) outs(%[[VAL_35]] : tensor<128xi64>) {
// CHECK:               ^bb0(%[[VAL_99:.*]]: i64, %[[VAL_100:.*]]: i64, %[[VAL_101:.*]]: i64):
// CHECK:                 %[[VAL_102:.*]] = arith.addi %[[VAL_99]], %[[VAL_100]] : i64
// CHECK:                 linalg.yield %[[VAL_102]] : i64
// CHECK:               } -> tensor<128xi64>
// CHECK:               %[[VAL_103:.*]] = arith.addi %[[VAL_31]], %[[VAL_20]] : index
// CHECK:               %[[VAL_104:.*]] = arith.addi %[[VAL_68]], %[[VAL_20]] : index
// CHECK:               scf.yield %[[VAL_97]], %[[VAL_98]], %[[VAL_103]], %[[VAL_19]], %[[VAL_104]] : i32, tensor<128xi64>, index, index, index
// CHECK:             }
// CHECK:             %[[VAL_105:.*]] = arith.index_cast %[[VAL_106:.*]]#4 : index to i32
// CHECK:             %[[VAL_107:.*]] = linalg.fill ins(%[[VAL_105]] : i32) outs(%[[VAL_24]] : tensor<128xi32>) -> tensor<128xi32>
// CHECK:             %[[VAL_108:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_27]], %[[VAL_107]] : tensor<128xi32>, tensor<128xi32>) outs(%[[VAL_27]] : tensor<128xi32>) {
// CHECK:             ^bb0(%[[VAL_109:.*]]: i32, %[[VAL_110:.*]]: i32, %[[VAL_111:.*]]: i32):
// CHECK:               %[[VAL_112:.*]] = arith.addi %[[VAL_109]], %[[VAL_110]] : i32
// CHECK:               linalg.yield %[[VAL_112]] : i32
// CHECK:             } -> tensor<128xi32>
// CHECK:             %[[VAL_113:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_106]]#1, %[[VAL_23]] : tensor<128xi64>, tensor<128xi64>) outs(%[[VAL_106]]#1 : tensor<128xi64>) {
// CHECK:             ^bb0(%[[VAL_114:.*]]: i64, %[[VAL_115:.*]]: i64, %[[VAL_116:.*]]: i64):
// CHECK:               %[[VAL_117:.*]] = arith.addi %[[VAL_114]], %[[VAL_115]] : i64
// CHECK:               linalg.yield %[[VAL_117]] : i64
// CHECK:             } -> tensor<128xi64>
// CHECK:             %[[VAL_118:.*]] = arith.addi %[[VAL_106]]#4, %[[VAL_20]] : index
// CHECK:             %[[VAL_119:.*]] = arith.addi %[[VAL_106]]#2, %[[VAL_20]] : index
// CHECK:             scf.yield %[[VAL_108]], %[[VAL_113]], %[[VAL_118]], %[[VAL_119]], %[[VAL_106]]#3 : tensor<128xi32>, tensor<128xi64>, index, index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
