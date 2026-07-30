// RUN: triton-shared-opt --triton-to-linalg-experimental --cse %s | FileCheck %s

// RUN: triton-shared-opt --triton-to-linalg-experimental="enable-make-gather-scatter=false" %s | FileCheck %s --check-prefix=NO_GATHER

// Make sure no extract_slice when enable-make-gather-scatter=false..
// NO_GATHER-LABLE: func.func @row_gather2(
// NO_GATHER-NOT: extract_slice


// Make sure extract_slice is generated correctly.

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:   func.func @row_gather2(
// CHECK-SAME:                           %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:                           %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<*xf32>,
// CHECK-SAME:                           %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i64,
// CHECK-SAME:                           %[[VAL_13:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_14:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_15:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_16:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_17:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                           %[[VAL_18:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) {
// CHECK:           %[[VAL_19:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_20:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_21:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_22:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_23:.*]] = tensor.empty() : tensor<16x1xi32>
// CHECK:           %[[VAL_24:.*]] = linalg.fill ins(%[[VAL_20]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_25:.*]] = arith.muli %[[VAL_5]], %[[VAL_20]] : i32
// CHECK:           %[[VAL_26:.*]] = tensor.empty() : tensor<16xi32>
// CHECK:           %[[VAL_27:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]]], iterator_types = ["parallel"]} outs(%[[VAL_26]] : tensor<16xi32>) {
// CHECK:           ^bb0(%[[VAL_28:.*]]: i32):
// CHECK:             %[[VAL_29:.*]] = linalg.index 0 : index
// CHECK:             %[[VAL_30:.*]] = arith.index_cast %[[VAL_29]] : index to i32
// CHECK:             linalg.yield %[[VAL_30]] : i32
// CHECK:           } -> tensor<16xi32>
// CHECK:           %[[VAL_31:.*]] = tensor.expand_shape %[[VAL_27]] {{\[\[}}0, 1]] output_shape [16, 1] : tensor<16xi32> into tensor<16x1xi32>
// CHECK:           %[[VAL_32:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_31]], %[[VAL_24]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_31]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_33:.*]]: i32, %[[VAL_34:.*]]: i32, %[[VAL_35:.*]]: i32):
// CHECK:             %[[VAL_36:.*]] = arith.remsi %[[VAL_33]], %[[VAL_34]] : i32
// CHECK:             linalg.yield %[[VAL_36]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_37:.*]] = arith.muli %[[VAL_3]], %[[VAL_20]] : i32
// CHECK:           %[[VAL_38:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_31]], %[[VAL_24]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_31]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_39:.*]]: i32, %[[VAL_40:.*]]: i32, %[[VAL_41:.*]]: i32):
// CHECK:             %[[VAL_42:.*]] = arith.divsi %[[VAL_39]], %[[VAL_40]] : i32
// CHECK:             linalg.yield %[[VAL_42]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_43:.*]] = arith.extsi %[[VAL_2]] : i32 to i64
// CHECK:           %[[VAL_44:.*]] = arith.muli %[[VAL_43]], %[[VAL_7]] : i64
// CHECK:           %[[VAL_45:.*]] = arith.index_cast %[[VAL_44]] : i64 to index
// CHECK:           %[[VAL_46:.*]] = arith.index_cast %[[VAL_8]] : i64 to index
// CHECK:           %[[VAL_47:.*]] = linalg.fill ins(%[[VAL_4]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_48:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_38]], %[[VAL_47]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_38]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_49:.*]]: i32, %[[VAL_50:.*]]: i32, %[[VAL_51:.*]]: i32):
// CHECK:             %[[VAL_52:.*]] = arith.addi %[[VAL_49]], %[[VAL_50]] : i32
// CHECK:             linalg.yield %[[VAL_52]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_53:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK:           %[[VAL_54:.*]] = linalg.fill ins(%[[VAL_37]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_55:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_32]], %[[VAL_54]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_32]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_56:.*]]: i32, %[[VAL_57:.*]]: i32, %[[VAL_58:.*]]: i32):
// CHECK:             %[[VAL_59:.*]] = arith.addi %[[VAL_56]], %[[VAL_57]] : i32
// CHECK:             linalg.yield %[[VAL_59]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_60:.*]] = arith.index_cast %[[VAL_53]] : index to i32
// CHECK:           %[[VAL_61:.*]] = linalg.fill ins(%[[VAL_60]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_62:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_48]], %[[VAL_61]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_48]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_63:.*]]: i32, %[[VAL_64:.*]]: i32, %[[VAL_65:.*]]: i32):
// CHECK:             %[[VAL_66:.*]] = arith.muli %[[VAL_63]], %[[VAL_64]] : i32
// CHECK:             linalg.yield %[[VAL_66]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_67:.*]] = arith.index_cast %[[VAL_46]] : index to i32
// CHECK:           %[[VAL_68:.*]] = linalg.fill ins(%[[VAL_67]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_69:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_55]], %[[VAL_68]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_55]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_70:.*]]: i32, %[[VAL_71:.*]]: i32, %[[VAL_72:.*]]: i32):
// CHECK:             %[[VAL_73:.*]] = arith.muli %[[VAL_70]], %[[VAL_71]] : i32
// CHECK:             linalg.yield %[[VAL_73]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_74:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_62]], %[[VAL_69]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_62]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_75:.*]]: i32, %[[VAL_76:.*]]: i32, %[[VAL_77:.*]]: i32):
// CHECK:             %[[VAL_78:.*]] = arith.addi %[[VAL_75]], %[[VAL_76]] : i32
// CHECK:             linalg.yield %[[VAL_78]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_79:.*]] = arith.index_cast %[[VAL_45]] : index to i32
// CHECK:           %[[VAL_80:.*]] = linalg.fill ins(%[[VAL_79]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_81:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_80]], %[[VAL_74]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_80]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_82:.*]]: i32, %[[VAL_83:.*]]: i32, %[[VAL_84:.*]]: i32):
// CHECK:             %[[VAL_85:.*]] = arith.addi %[[VAL_82]], %[[VAL_83]] : i32
// CHECK:             linalg.yield %[[VAL_85]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_86:.*]] = tensor.collapse_shape %[[VAL_81]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           %[[VAL_87:.*]] = arith.index_cast %[[VAL_25]] : i32 to index
// CHECK:           %[[VAL_88:.*]] = arith.minsi %[[VAL_87]], %[[VAL_21]] : index
// CHECK:           %[[VAL_89:.*]] = arith.maxsi %[[VAL_88]], %[[VAL_22]] : index
// CHECK:           %[[VAL_90:.*]] = arith.minsi %[[VAL_89]], %[[VAL_21]] : index
// CHECK:           %[[VAL_91:.*]] = memref.alloc() : memref<16x16xf32>
// CHECK:           %[[VAL_92:.*]] = arith.minsi %[[VAL_90]], %[[VAL_21]] : index
// CHECK:           scf.for %[[VAL_93:.*]] = %[[VAL_22]] to %[[VAL_92]] step %[[VAL_19]] {
// CHECK:             %[[VAL_94:.*]] = tensor.extract %[[VAL_86]]{{\[}}%[[VAL_93]]] : tensor<16xi32>
// CHECK:             %[[VAL_95:.*]] = arith.index_cast %[[VAL_94]] : i32 to index
// CHECK:             %[[VAL_96:.*]] = memref.reinterpret_cast %[[VAL_0]] to offset: {{\[}}%[[VAL_95]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_97:.*]] = memref.subview %[[VAL_96]][0, 0] [1, 8] [1, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_98:.*]] = memref.subview %[[VAL_91]]{{\[}}%[[VAL_93]], 0] [1, 8] [1, 1] : memref<16x16xf32> to memref<1x8xf32, strided<[16, 1], offset: ?>>
// CHECK:             memref.copy %[[VAL_97]], %[[VAL_98]] : memref<1x8xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[16, 1], offset: ?>>
// CHECK:           }
// CHECK:           %[[VAL_99:.*]] = bufferization.to_tensor %[[VAL_91]] restrict writable : memref<16x16xf32> to tensor<16x16xf32>
// CHECK:           %[[VAL_100:.*]] = arith.muli %[[VAL_43]], %[[VAL_10]] : i64
// CHECK:           %[[VAL_101:.*]] = arith.extsi %[[VAL_18]] : i32 to i64
// CHECK:           %[[VAL_102:.*]] = arith.muli %[[VAL_101]], %[[VAL_12]] : i64
// CHECK:           %[[VAL_103:.*]] = arith.addi %[[VAL_100]], %[[VAL_102]] : i64
// CHECK:           %[[VAL_104:.*]] = arith.index_cast %[[VAL_103]] : i64 to index
// CHECK:           %[[VAL_105:.*]] = arith.index_cast %[[VAL_11]] : i64 to index
// CHECK:           %[[VAL_106:.*]] = arith.index_cast %[[VAL_9]] : i64 to index
// CHECK:           %[[VAL_107:.*]] = arith.index_cast %[[VAL_106]] : index to i32
// CHECK:           %[[VAL_108:.*]] = linalg.fill ins(%[[VAL_107]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_109:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_48]], %[[VAL_108]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_48]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_110:.*]]: i32, %[[VAL_111:.*]]: i32, %[[VAL_112:.*]]: i32):
// CHECK:             %[[VAL_113:.*]] = arith.muli %[[VAL_110]], %[[VAL_111]] : i32
// CHECK:             linalg.yield %[[VAL_113]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_114:.*]] = arith.index_cast %[[VAL_105]] : index to i32
// CHECK:           %[[VAL_115:.*]] = linalg.fill ins(%[[VAL_114]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_116:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_55]], %[[VAL_115]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_55]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_117:.*]]: i32, %[[VAL_118:.*]]: i32, %[[VAL_119:.*]]: i32):
// CHECK:             %[[VAL_120:.*]] = arith.muli %[[VAL_117]], %[[VAL_118]] : i32
// CHECK:             linalg.yield %[[VAL_120]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_121:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_109]], %[[VAL_116]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_109]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_122:.*]]: i32, %[[VAL_123:.*]]: i32, %[[VAL_124:.*]]: i32):
// CHECK:             %[[VAL_125:.*]] = arith.addi %[[VAL_122]], %[[VAL_123]] : i32
// CHECK:             linalg.yield %[[VAL_125]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_126:.*]] = arith.index_cast %[[VAL_104]] : index to i32
// CHECK:           %[[VAL_127:.*]] = linalg.fill ins(%[[VAL_126]] : i32) outs(%[[VAL_23]] : tensor<16x1xi32>) -> tensor<16x1xi32>
// CHECK:           %[[VAL_128:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_1]], #[[$ATTR_1]], #[[$ATTR_1]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_127]], %[[VAL_121]] : tensor<16x1xi32>, tensor<16x1xi32>) outs(%[[VAL_127]] : tensor<16x1xi32>) {
// CHECK:           ^bb0(%[[VAL_129:.*]]: i32, %[[VAL_130:.*]]: i32, %[[VAL_131:.*]]: i32):
// CHECK:             %[[VAL_132:.*]] = arith.addi %[[VAL_129]], %[[VAL_130]] : i32
// CHECK:             linalg.yield %[[VAL_132]] : i32
// CHECK:           } -> tensor<16x1xi32>
// CHECK:           %[[VAL_133:.*]] = tensor.collapse_shape %[[VAL_128]] {{\[\[}}0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:           scf.for %[[VAL_134:.*]] = %[[VAL_22]] to %[[VAL_92]] step %[[VAL_19]] {
// CHECK:             %[[VAL_135:.*]] = tensor.extract %[[VAL_133]]{{\[}}%[[VAL_134]]] : tensor<16xi32>
// CHECK:             %[[VAL_136:.*]] = arith.index_cast %[[VAL_135]] : i32 to index
// CHECK:             %[[VAL_137:.*]] = tensor.extract_slice %[[VAL_99]]{{\[}}%[[VAL_134]], 0] [1, 8] [1, 1] : tensor<16x16xf32> to tensor<1x8xf32>
// CHECK:             %[[VAL_138:.*]] = memref.reinterpret_cast %[[VAL_1]] to offset: {{\[}}%[[VAL_136]]], sizes: [1, 16], strides: [1, 1] : memref<*xf32> to memref<1x16xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_139:.*]] = memref.subview %[[VAL_138]][0, 0] [1, 8] [1, 1] : memref<1x16xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[1, 1], offset: ?>>
// CHECK:             %[[VAL_140:.*]] = memref.cast %[[VAL_139]] : memref<1x8xf32, strided<[1, 1], offset: ?>> to memref<1x8xf32, strided<[?, ?], offset: ?>>
// CHECK:             bufferization.materialize_in_destination %[[VAL_137]] in writable %[[VAL_140]] : (tensor<1x8xf32>, memref<1x8xf32, strided<[?, ?], offset: ?>>) -> ()
// CHECK:           }
// CHECK:           return

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