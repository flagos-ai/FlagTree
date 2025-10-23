// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func public @matmul_kernel_with_block_pointers_01234567891011(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32) {
    %c64_i32 = arith.constant 64 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : bf16
    %c256_i32 = arith.constant 256 : i32
    %0 = arith.extsi %arg3 : i32 to i64
    %1 = arith.extsi %arg5 : i32 to i64
    %2 = arith.extsi %arg6 : i32 to i64
    %3 = arith.extsi %arg7 : i32 to i64
    %4 = tt.make_tensor_ptr %arg0, [%0, %1], [%2, %3], [%arg12, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    %5 = tt.advance %4, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
    %6 = tt.splat %cst : bf16 -> tensor<128x64xbf16>
    %7:3 = scf.for %arg14 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg15 = %6, %arg16 = %5, %arg17 = %4) -> (tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>)  : i32 {
      %13 = tt.load %arg16 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>>
      %14 = tt.load %arg17 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<tensor<128x64xbf16>>
      %15 = arith.addf %13, %14 : tensor<128x64xbf16>
      %16 = arith.addf %arg15, %15 : tensor<128x64xbf16>
      %17 = tt.advance %arg16, [%c0_i32, %c64_i32] : <tensor<128x64xbf16>>
      %18 = tt.advance %arg17, [%c64_i32, %c0_i32] : <tensor<128x64xbf16>>
      scf.yield %16, %17, %18 : tensor<128x64xbf16>, !tt.ptr<tensor<128x64xbf16>>, !tt.ptr<tensor<128x64xbf16>>
    }
    %8 = arith.extsi %arg10 : i32 to i64
    %9 = arith.extsi %arg11 : i32 to i64
    %10 = arith.extsi %arg4 : i32 to i64
    %11 = arith.muli %arg13, %c256_i32 : i32
    %12 = tt.make_tensor_ptr %arg2, [%0, %10], [%8, %9], [%arg12, %11] {order = array<i32: 1, 0>} : <tensor<128x64xbf16>>
    tt.store %12, %7#0 {boundaryCheck = array<i32: 0, 1>, cache = 1 : i32, evict = 1 : i32} : !tt.ptr<tensor<128x64xbf16>>
    tt.return
  }
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-LABEL:   func.func @matmul_kernel_with_block_pointers_01234567891011(
// CHECK-SAME:        %[[VAL_0:.*]]: memref<?xi8>, %[[VAL_1:.*]]: memref<?xi8>, %[[VAL_2:.*]]: memref<?xbf16>, %[[VAL_3:.*]]: memref<?xbf16>, %[[VAL_4:.*]]: memref<?xbf16> {tt.tensor_kind = 1 : i32},
// CHECK-SAME:        %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32, %[[VAL_11:.*]]: i32, %[[VAL_12:.*]]: i32, %[[VAL_13:.*]]: i32, %[[VAL_14:.*]]: i32, %[[VAL_15:.*]]: i32, %[[VAL_16:.*]]: i32, %[[VAL_17:.*]]: i32, %[[VAL_18:.*]]: i32, %[[VAL_19:.*]]: i32, %[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:           %[[VAL_22:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_23:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_24:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_25:.*]] = arith.constant 256 : i32
// CHECK:           %[[VAL_26:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_27:.*]] = arith.constant 64 : i32
// CHECK:           %[[VAL_28:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK:           %[[VAL_29:.*]] = tensor.empty() : tensor<128x64xbf16>
// CHECK:           %[[VAL_30:.*]] = linalg.fill ins(%[[VAL_28]] : bf16) outs(%[[VAL_29]] : tensor<128x64xbf16>) -> tensor<128x64xbf16>
// CHECK:           %[[VAL_31:.*]] = arith.index_cast %[[VAL_14]] : i32 to index
// CHECK:           %[[VAL_32:.*]] = arith.index_cast %[[VAL_8]] : i32 to index
// CHECK:           %[[VAL_33:.*]] = arith.index_cast %[[VAL_9]] : i32 to index
// CHECK:           %[[VAL_34:.*]] = arith.muli %[[VAL_31]], %[[VAL_32]] : index
// CHECK:           %[[VAL_35:.*]] = arith.index_cast %[[VAL_5]] : i32 to index
// CHECK:           %[[VAL_36:.*]] = arith.index_cast %[[VAL_7]] : i32 to index
// CHECK:           %[[VAL_37:.*]] = arith.muli %[[VAL_33]], %[[VAL_24]] : index
// CHECK:           %[[VAL_38:.*]] = arith.addi %[[VAL_34]], %[[VAL_37]] : index
// CHECK:           %[[VAL_39:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_38]]], sizes: [128, 64], strides: {{\[}}%[[VAL_32]], %[[VAL_33]]] : memref<?xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_40:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_34]]], sizes: [128, 64], strides: {{\[}}%[[VAL_32]], %[[VAL_33]]] : memref<?xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_41:.*]]:7 = scf.for %[[VAL_42:.*]] = %[[VAL_26]] to %[[VAL_7]] step %[[VAL_27]] iter_args(%[[VAL_43:.*]] = %[[VAL_30]], %[[VAL_44:.*]] = %[[VAL_39]], %[[VAL_45:.*]] = %[[VAL_40]], %[[VAL_46:.*]] = %[[VAL_38]], %[[VAL_47:.*]] = %[[VAL_23]], %[[VAL_48:.*]] = %[[VAL_34]], %[[VAL_49:.*]] = %[[VAL_23]]) -> (tensor<128x64xbf16>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, index, index, index, index)  : i32 {
// CHECK:             %[[VAL_50:.*]] = memref.alloc() : memref<128x64xbf16>
// CHECK:             %[[VAL_51:.*]] = arith.divsi %[[VAL_46]], %[[VAL_32]] : index
// CHECK:             %[[VAL_52:.*]] = arith.subi %[[VAL_35]], %[[VAL_51]] : index
// CHECK:             %[[VAL_53:.*]] = arith.maxsi %[[VAL_52]], %[[VAL_23]] : index
// CHECK:             %[[VAL_54:.*]] = arith.minsi %[[VAL_53]], %[[VAL_22]] : index
// CHECK:             %[[VAL_55:.*]] = arith.remsi %[[VAL_46]], %[[VAL_32]] : index
// CHECK:             %[[VAL_56:.*]] = arith.divsi %[[VAL_55]], %[[VAL_33]] : index
// CHECK:             %[[VAL_57:.*]] = arith.subi %[[VAL_36]], %[[VAL_56]] : index
// CHECK:             %[[VAL_58:.*]] = arith.maxsi %[[VAL_57]], %[[VAL_23]] : index
// CHECK:             %[[VAL_59:.*]] = arith.minsi %[[VAL_58]], %[[VAL_24]] : index
// CHECK:             %[[VAL_60:.*]] = memref.subview %[[VAL_44]][0, 0] {{\[}}%[[VAL_54]], %[[VAL_59]]] [1, 1] : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
// CHECK:             %[[VAL_61:.*]] = memref.subview %[[VAL_50]][0, 0] {{\[}}%[[VAL_54]], %[[VAL_59]]] [1, 1] : memref<128x64xbf16> to memref<?x?xbf16, strided<[64, 1]>>
// CHECK:             memref.copy %[[VAL_60]], %[[VAL_61]] : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[64, 1]>>
// CHECK:             %[[VAL_62:.*]] = bufferization.to_tensor %[[VAL_50]] restrict writable : memref<128x64xbf16>
// CHECK:             %[[VAL_63:.*]] = memref.alloc() : memref<128x64xbf16>
// CHECK:             %[[VAL_64:.*]] = arith.divsi %[[VAL_48]], %[[VAL_32]] : index
// CHECK:             %[[VAL_65:.*]] = arith.subi %[[VAL_35]], %[[VAL_64]] : index
// CHECK:             %[[VAL_66:.*]] = arith.maxsi %[[VAL_65]], %[[VAL_23]] : index
// CHECK:             %[[VAL_67:.*]] = arith.minsi %[[VAL_66]], %[[VAL_22]] : index
// CHECK:             %[[VAL_68:.*]] = arith.remsi %[[VAL_48]], %[[VAL_32]] : index
// CHECK:             %[[VAL_69:.*]] = arith.divsi %[[VAL_68]], %[[VAL_33]] : index
// CHECK:             %[[VAL_70:.*]] = arith.subi %[[VAL_36]], %[[VAL_69]] : index
// CHECK:             %[[VAL_71:.*]] = arith.maxsi %[[VAL_70]], %[[VAL_23]] : index
// CHECK:             %[[VAL_72:.*]] = arith.minsi %[[VAL_71]], %[[VAL_24]] : index
// CHECK:             %[[VAL_73:.*]] = memref.subview %[[VAL_45]][0, 0] {{\[}}%[[VAL_67]], %[[VAL_72]]] [1, 1] : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
// CHECK:             %[[VAL_74:.*]] = memref.subview %[[VAL_63]][0, 0] {{\[}}%[[VAL_67]], %[[VAL_72]]] [1, 1] : memref<128x64xbf16> to memref<?x?xbf16, strided<[64, 1]>>
// CHECK:             memref.copy %[[VAL_73]], %[[VAL_74]] : memref<?x?xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[64, 1]>>
// CHECK:             %[[VAL_75:.*]] = bufferization.to_tensor %[[VAL_63]] restrict writable : memref<128x64xbf16>
// CHECK:             %[[VAL_76:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_62]], %[[VAL_75]] : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%[[VAL_62]] : tensor<128x64xbf16>) {
// CHECK:             ^bb0(%[[VAL_77:.*]]: bf16, %[[VAL_78:.*]]: bf16, %[[VAL_79:.*]]: bf16):
// CHECK:               %[[VAL_80:.*]] = arith.addf %[[VAL_77]], %[[VAL_78]] : bf16
// CHECK:               linalg.yield %[[VAL_80]] : bf16
// CHECK:             } -> tensor<128x64xbf16>
// CHECK:             %[[VAL_81:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel", "parallel"]} ins(%[[VAL_43]], %[[VAL_76]] : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%[[VAL_43]] : tensor<128x64xbf16>) {
// CHECK:             ^bb0(%[[VAL_82:.*]]: bf16, %[[VAL_83:.*]]: bf16, %[[VAL_84:.*]]: bf16):
// CHECK:               %[[VAL_85:.*]] = arith.addf %[[VAL_82]], %[[VAL_83]] : bf16
// CHECK:               linalg.yield %[[VAL_85]] : bf16
// CHECK:             } -> tensor<128x64xbf16>
// CHECK:             %[[VAL_86:.*]] = arith.muli %[[VAL_33]], %[[VAL_24]] : index
// CHECK:             %[[VAL_87:.*]] = arith.addi %[[VAL_86]], %[[VAL_47]] : index
// CHECK:             %[[VAL_88:.*]] = arith.addi %[[VAL_46]], %[[VAL_87]] : index
// CHECK:             %[[VAL_89:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_88]]], sizes: [128, 64], strides: {{\[}}%[[VAL_32]], %[[VAL_33]]] : memref<?xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:             %[[VAL_90:.*]] = arith.muli %[[VAL_32]], %[[VAL_24]] : index
// CHECK:             %[[VAL_91:.*]] = arith.addi %[[VAL_90]], %[[VAL_48]] : index
// CHECK:             %[[VAL_92:.*]] = arith.addi %[[VAL_91]], %[[VAL_49]] : index
// CHECK:             %[[VAL_93:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_92]]], sizes: [128, 64], strides: {{\[}}%[[VAL_32]], %[[VAL_33]]] : memref<?xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:             scf.yield %[[VAL_81]], %[[VAL_89]], %[[VAL_93]], %[[VAL_88]], %[[VAL_23]], %[[VAL_92]], %[[VAL_23]] : tensor<128x64xbf16>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, index, index, index, index
// CHECK:           }
// CHECK:           %[[VAL_94:.*]] = arith.muli %[[VAL_15]], %[[VAL_25]] : i32
// CHECK:           %[[VAL_95:.*]] = arith.index_cast %[[VAL_94]] : i32 to index
// CHECK:           %[[VAL_96:.*]] = arith.index_cast %[[VAL_12]] : i32 to index
// CHECK:           %[[VAL_97:.*]] = arith.index_cast %[[VAL_13]] : i32 to index
// CHECK:           %[[VAL_98:.*]] = arith.muli %[[VAL_31]], %[[VAL_96]] : index
// CHECK:           %[[VAL_99:.*]] = arith.muli %[[VAL_95]], %[[VAL_97]] : index
// CHECK:           %[[VAL_100:.*]] = arith.index_cast %[[VAL_6]] : i32 to index
// CHECK:           %[[VAL_101:.*]] = arith.addi %[[VAL_98]], %[[VAL_99]] : index
// CHECK:           %[[VAL_102:.*]] = memref.reinterpret_cast %[[VAL_4]] to offset: {{\[}}%[[VAL_101]]], sizes: [128, 64], strides: {{\[}}%[[VAL_96]], %[[VAL_97]]] : memref<?xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:           %[[VAL_103:.*]] = arith.divsi %[[VAL_101]], %[[VAL_96]] : index
// CHECK:           %[[VAL_104:.*]] = arith.subi %[[VAL_35]], %[[VAL_103]] : index
// CHECK:           %[[VAL_105:.*]] = arith.maxsi %[[VAL_104]], %[[VAL_23]] : index
// CHECK:           %[[VAL_106:.*]] = arith.minsi %[[VAL_105]], %[[VAL_22]] : index
// CHECK:           %[[VAL_107:.*]] = arith.remsi %[[VAL_101]], %[[VAL_96]] : index
// CHECK:           %[[VAL_108:.*]] = arith.divsi %[[VAL_107]], %[[VAL_97]] : index
// CHECK:           %[[VAL_109:.*]] = arith.subi %[[VAL_100]], %[[VAL_108]] : index
// CHECK:           %[[VAL_110:.*]] = arith.maxsi %[[VAL_109]], %[[VAL_23]] : index
// CHECK:           %[[VAL_111:.*]] = arith.minsi %[[VAL_110]], %[[VAL_24]] : index
// CHECK:           %[[VAL_112:.*]] = tensor.extract_slice %[[VAL_113:.*]]#0[0, 0] {{\[}}%[[VAL_106]], %[[VAL_111]]] [1, 1] : tensor<128x64xbf16> to tensor<?x?xbf16>
// CHECK:           %[[VAL_114:.*]] = memref.subview %[[VAL_102]][0, 0] {{\[}}%[[VAL_106]], %[[VAL_111]]] [1, 1] : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<?x?xbf16, strided<[?, ?], offset: ?>>
// CHECK:           bufferization.materialize_in_destination %[[VAL_112]] in writable %[[VAL_114]] : (tensor<?x?xbf16>, memref<?x?xbf16, strided<[?, ?], offset: ?>>) -> ()
// CHECK:           return
// CHECK:         }

