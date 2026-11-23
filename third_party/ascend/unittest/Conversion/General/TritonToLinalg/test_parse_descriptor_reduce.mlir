// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @kernel
// CHECK: %[[SUBVIEW:.*]] = memref.subview %[[RC:.*]][%[[VAL_36:.*]], %[[VAL_39:.*]]] [%[[VAL_38:.*]], %[[VAL_41:.*]]] [1, 1] : memref<2x16xi32, strided<[?, 1], offset: ?>> to memref<?x?xi32, strided<[?, 1], offset: ?>>
// CHECK: %[[EXTRACT:.*]] = tensor.extract_slice %[[VAL_7:.*]][%[[VAL_36]], %[[VAL_39]]] [%[[VAL_38]], %[[VAL_41]]] [1, 1] : tensor<2x16xi32> to tensor<?x?xi32>
// CHECK: hivm.hir.store ins(%[[EXTRACT]] : tensor<?x?xi32>) outs(%[[SUBVIEW]] : memref<?x?xi32, strided<[?, 1], offset: ?>>) atomic = <add>

module {
  tt.func public @kernel(%a_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %out_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %M: i32 , %N: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %desc = arith.constant 1 : i64
    %c16_i32 = arith.constant 16 : i32
    %c2_i32 = arith.constant 2 : i32
    %moffset = tt.get_program_id x : i32
    %moffset_0 = arith.muli %moffset, %c2_i32 : i32
    %noffset = tt.get_program_id y : i32
    %noffset_1 = arith.muli %noffset, %c16_i32 : i32
    %midx = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %midx_2 = tt.expand_dims %midx {axis = 1 : i32} : tensor<2xi32> -> tensor<2x1xi32>
    %midx_3 = tt.splat %moffset_0 : i32 -> tensor<2x1xi32>
    %midx_4 = arith.addi %midx_3, %midx_2 : tensor<2x1xi32>
    %nidx = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %nidx_5 = tt.expand_dims %nidx {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %nidx_6 = tt.splat %noffset_1 : i32 -> tensor<1x16xi32>
    %nidx_7 = arith.addi %nidx_6, %nidx_5 : tensor<1x16xi32>
    %idx = tt.splat %N : i32 -> tensor<2x1xi32>
    %idx_8 = arith.muli %midx_4, %idx : tensor<2x1xi32>
    %idx_9 = tt.broadcast %idx_8 : tensor<2x1xi32> -> tensor<2x16xi32>
    %idx_10 = tt.broadcast %nidx_7 : tensor<1x16xi32> -> tensor<2x16xi32>
    %idx_11 = arith.addi %idx_9, %idx_10 : tensor<2x16xi32>
    %val = tt.splat %a_ptr : !tt.ptr<i32> -> tensor<2x16x!tt.ptr<i32>>
    %val_12 = tt.addptr %val, %idx_11 : tensor<2x16x!tt.ptr<i32>>, tensor<2x16xi32>
    %val_13 = tt.load %val_12 : tensor<2x16x!tt.ptr<i32>>
    %desc_14 = arith.extsi %N : i32 to i64
    %desc_15 = tt.make_tensor_descriptor %out_ptr, [%M, %N], [%desc_14, %desc] : <i32>, <tensor<2x16xsi32>>
    tt.descriptor_reduce add, %desc_15[%moffset_0, %noffset_1], %val_13 : !tt.tensordesc<tensor<2x16xsi32>>, tensor<2x16xi32>
    tt.return
  }
}
