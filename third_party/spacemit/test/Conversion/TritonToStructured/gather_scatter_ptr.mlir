// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --cse --canonicalize %s | FileCheck %s

// RUN: triton-shared-opt --triton-to-structured="enable-make-gather-scatter=false" --remove-dead-values --canonicalize %s | FileCheck %s --check-prefix=NO_GATHER

// Make sure no tts.make_gather_scatter_tptr generated when enable-make-gather-scatter=false.
// NO_GATHER-LABEL:   tt.func public @row_gather2(
// NO_GATHER-NOT:     tts.make_gather_scatter_tptr

// Make sure tts.make_gather_scatter_tptr is generated with correct gather_dim stride and offset.

// CHECK-LABEL:   tt.func public @row_gather2(
// CHECK-SAME:                                %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                                %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                                %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                                %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                %[[VAL_9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                %[[VAL_10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                %[[VAL_11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                                %[[VAL_12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64) attributes {noinline = false} {
// CHECK:           %[[VAL_13:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_14:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_15:.*]] = arith.constant dense<8> : tensor<16x1xi32>
// CHECK:           %[[VAL_16:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_17:.*]] = tt.get_program_id z : i32
// CHECK:           %[[VAL_18:.*]] = arith.muli %[[VAL_5]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_19:.*]] = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
// CHECK:           %[[VAL_20:.*]] = tt.expand_dims %[[VAL_19]] {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
// CHECK:           %[[VAL_21:.*]] = arith.remsi %[[VAL_20]], %[[VAL_15]] : tensor<16x1xi32>
// CHECK:           %[[VAL_22:.*]] = arith.muli %[[VAL_3]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_23:.*]] = arith.divsi %[[VAL_20]], %[[VAL_15]] : tensor<16x1xi32>
// CHECK:           %[[VAL_24:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_25:.*]] = arith.muli %[[VAL_24]], %[[VAL_7]] : i64
// CHECK:           %[[VAL_26:.*]] = arith.index_cast %[[VAL_25]] : i64 to index
// CHECK:           %[[VAL_27:.*]] = tt.splat %[[VAL_4]] : i32 -> tensor<16x1xi32>
// CHECK:           %[[VAL_28:.*]] = arith.addi %[[VAL_23]], %[[VAL_27]] : tensor<16x1xi32>
// CHECK:           %[[VAL_29:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK:           %[[VAL_30:.*]] = tt.splat %[[VAL_22]] : i32 -> tensor<16x1xi32>
// CHECK:           %[[VAL_31:.*]] = arith.addi %[[VAL_21]], %[[VAL_30]] : tensor<16x1xi32>
// CHECK:           %[[VAL_32:.*]] = arith.index_cast %[[VAL_8]] : i64 to index
// CHECK:           %[[VAL_33:.*]] = arith.index_cast %[[VAL_29]] : index to i32
// CHECK:           %[[VAL_34:.*]] = tt.splat %[[VAL_33]] : i32 -> tensor<16x1xi32>
// CHECK:           %[[VAL_35:.*]] = arith.muli %[[VAL_28]], %[[VAL_34]] : tensor<16x1xi32>
// CHECK:           %[[VAL_36:.*]] = arith.index_cast %[[VAL_32]] : index to i32
// CHECK:           %[[VAL_37:.*]] = tt.splat %[[VAL_36]] : i32 -> tensor<16x1xi32>
// CHECK:           %[[VAL_38:.*]] = arith.muli %[[VAL_31]], %[[VAL_37]] : tensor<16x1xi32>
// CHECK:           %[[VAL_39:.*]] = arith.addi %[[VAL_35]], %[[VAL_38]] : tensor<16x1xi32>
// CHECK:           %[[VAL_40:.*]] = arith.index_cast %[[VAL_26]] : index to i32
// CHECK:           %[[VAL_41:.*]] = tt.splat %[[VAL_40]] : i32 -> tensor<16x1xi32>
// CHECK:           %[[VAL_42:.*]] = arith.addi %[[VAL_41]], %[[VAL_39]] : tensor<16x1xi32>
// CHECK:           %[[VAL_43:.*]] = tensor.collapse_shape %[[VAL_42]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           %[[VAL_44:.*]] = tts.make_gather_scatter_tptr %[[VAL_0]] to sizes: [16, 16] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_43]], strides: [1, 1], offsets: [0, 0] : tensor<16xi32> <f32> to !tt.ptr<tensor<16x16xf32>>
// CHECK:           %[[VAL_45:.*]] = arith.index_cast %[[VAL_18]] : i32 to index
// CHECK:           %[[VAL_46:.*]] = arith.minsi %[[VAL_45]], %[[VAL_14]] : index
// CHECK:           %[[VAL_47:.*]] = arith.maxsi %[[VAL_46]], %[[VAL_13]] : index
// CHECK:           %[[VAL_48:.*]] = arith.minsi %[[VAL_47]], %[[VAL_14]] : index
// CHECK:           %[[VAL_49:.*]] = "tts.load"(%[[VAL_44]], %[[VAL_48]]) <{operandSegmentSizes = array<i32: 1, 1, 0>, static_mask_dims = array<i64: -9223372036854775808, 8>}> : (!tt.ptr<tensor<16x16xf32>>, index) -> tensor<16x16xf32>
// CHECK:           %[[VAL_50:.*]] = arith.muli %[[VAL_24]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_51:.*]] = arith.extsi %[[VAL_17]] : i32 to i64
// CHECK:           %[[VAL_52:.*]] = arith.muli %[[VAL_51]], %[[VAL_12]] : i64
// CHECK:           %[[VAL_53:.*]] = arith.addi %[[VAL_50]], %[[VAL_52]] : i64
// CHECK:           %[[VAL_54:.*]] = arith.index_cast %[[VAL_53]] : i64 to index
// CHECK:           %[[VAL_55:.*]] = arith.index_cast %[[VAL_9]] : i64 to index
// CHECK:           %[[VAL_56:.*]] = arith.index_cast %[[VAL_11]] : i64 to index
// CHECK:           %[[VAL_57:.*]] = arith.index_cast %[[VAL_55]] : index to i32
// CHECK:           %[[VAL_58:.*]] = tt.splat %[[VAL_57]] : i32 -> tensor<16x1xi32>
// CHECK:           %[[VAL_59:.*]] = arith.muli %[[VAL_28]], %[[VAL_58]] : tensor<16x1xi32>
// CHECK:           %[[VAL_60:.*]] = arith.index_cast %[[VAL_56]] : index to i32
// CHECK:           %[[VAL_61:.*]] = tt.splat %[[VAL_60]] : i32 -> tensor<16x1xi32>
// CHECK:           %[[VAL_62:.*]] = arith.muli %[[VAL_31]], %[[VAL_61]] : tensor<16x1xi32>
// CHECK:           %[[VAL_63:.*]] = arith.addi %[[VAL_59]], %[[VAL_62]] : tensor<16x1xi32>
// CHECK:           %[[VAL_64:.*]] = arith.index_cast %[[VAL_54]] : index to i32
// CHECK:           %[[VAL_65:.*]] = tt.splat %[[VAL_64]] : i32 -> tensor<16x1xi32>
// CHECK:           %[[VAL_66:.*]] = arith.addi %[[VAL_65]], %[[VAL_63]] : tensor<16x1xi32>
// CHECK:           %[[VAL_67:.*]] = tensor.collapse_shape %[[VAL_66]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           %[[VAL_68:.*]] = tts.make_gather_scatter_tptr %[[VAL_1]] to sizes: [16, 16] gather_scatter_dim: 0 gather_scatter_offset: %[[VAL_67]], strides: [1, 1], offsets: [0, 0] : tensor<16xi32> <f32> to !tt.ptr<tensor<16x16xf32>>
// CHECK:           "tts.store"(%[[VAL_68]], %[[VAL_49]], %[[VAL_48]]) <{static_mask_dims = array<i64: -9223372036854775808, 8>}> : (!tt.ptr<tensor<16x16xf32>>, tensor<16x16xf32>, index) -> ()
// CHECK:           tt.return

module {
  tt.func public @row_gather2(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i64, %arg7: i64, %arg8: i64, %arg9: i64, %arg10: i64, %arg11: i64, %arg12: i64) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<1x16xi32>
    %cst_0 = arith.constant dense<8> : tensor<16x1xi32>
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id z : i32
    %1 = arith.muli %arg5, %c8_i32 : i32
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %3 = tt.expand_dims %2 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %4 = arith.remsi %3, %cst_0 : tensor<16x1xi32>
    %5 = arith.muli %arg3, %c8_i32 : i32
    %6 = tt.splat %5 : i32 -> tensor<16x1xi32>
    %7 = arith.addi %4, %6 : tensor<16x1xi32>
    %8 = arith.divsi %3, %cst_0 : tensor<16x1xi32>
    %9 = tt.splat %arg4 : i32 -> tensor<16x1xi32>
    %10 = arith.addi %8, %9 : tensor<16x1xi32>
    %11 = arith.extsi %arg2 : i32 to i64
    %12 = arith.muli %11, %arg7 : i64
    %13 = tt.addptr %arg0, %12 : !tt.ptr<f32>, i64
    %14 = arith.extsi %10 : tensor<16x1xi32> to tensor<16x1xi64>
    %15 = tt.splat %arg6 : i64 -> tensor<16x1xi64>
    %16 = arith.muli %14, %15 : tensor<16x1xi64>
    %17 = tt.splat %13 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %18 = arith.extsi %7 : tensor<16x1xi32> to tensor<16x1xi64>
    %19 = tt.splat %arg8 : i64 -> tensor<16x1xi64>
    %20 = arith.muli %18, %19 : tensor<16x1xi64>
    %21 = arith.addi %16, %20 : tensor<16x1xi64>
    %22 = tt.addptr %17, %21 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi64>
    %23 = tt.expand_dims %2 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %24 = tt.broadcast %22 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %25 = tt.broadcast %23 : tensor<1x16xi32> -> tensor<16x16xi32>
    %26 = tt.addptr %24, %25 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %27 = arith.cmpi slt, %23, %cst : tensor<1x16xi32>
    %28 = tt.splat %1 : i32 -> tensor<16x1xi32>
    %29 = arith.cmpi slt, %3, %28 : tensor<16x1xi32>
    %30 = tt.broadcast %27 : tensor<1x16xi1> -> tensor<16x16xi1>
    %31 = tt.broadcast %29 : tensor<16x1xi1> -> tensor<16x16xi1>
    %32 = arith.andi %30, %31 : tensor<16x16xi1>
    %33 = tt.load %26, %32 : tensor<16x16x!tt.ptr<f32>>
    %34 = arith.muli %11, %arg10 : i64
    %35 = arith.extsi %0 : i32 to i64
    %36 = arith.muli %35, %arg12 : i64
    %37 = arith.addi %34, %36 : i64
    %38 = tt.addptr %arg1, %37 : !tt.ptr<f32>, i64
    %39 = tt.splat %arg9 : i64 -> tensor<16x1xi64>
    %40 = arith.muli %14, %39 : tensor<16x1xi64>
    %41 = tt.splat %38 : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %42 = tt.splat %arg11 : i64 -> tensor<16x1xi64>
    %43 = arith.muli %18, %42 : tensor<16x1xi64>
    %44 = arith.addi %40, %43 : tensor<16x1xi64>
    %45 = tt.addptr %41, %44 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi64>
    %46 = tt.broadcast %45 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %47 = tt.addptr %46, %25 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    tt.store %47, %33, %32 : tensor<16x16x!tt.ptr<f32>>
    tt.return
  }
}