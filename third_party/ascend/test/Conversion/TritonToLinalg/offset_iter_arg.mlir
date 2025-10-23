// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @test_kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
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
    %7 = scf.for %arg2 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg3 = %4) -> (tensor<128xi32>)  : i32 {
      %8 = scf.for %arg4 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg5 = %arg3) -> (tensor<128xi32>)  : i32 {
        %10 = arith.cmpi slt, %arg5, %cst_1 : tensor<128xi32>
        %11 = tt.addptr %5, %arg5 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
        %12 = tt.load %11, %10, %cst_0 : tensor<128x!tt.ptr<i32>>
        %13 = tt.addptr %6, %arg5 : tensor<128x!tt.ptr<i32>>, tensor<128xi32>
        tt.store %13, %12, %10 : tensor<128x!tt.ptr<i32>>
        %14 = arith.addi %arg5, %cst : tensor<128xi32>
        scf.yield %14 : tensor<128xi32>
      }
      %9 = arith.addi %8, %cst : tensor<128xi32>
      scf.yield %9 : tensor<128xi32>
    }
    tt.return
  }
}

// CHECK-LABEL:   func.func @test_kernel(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<?xi8>, %[[VAL_1:.*]]: memref<?xi8>, 
// CHECK-SAME:                           %[[VAL_2:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, 
// CHECK-SAME:                           %[[VAL_3:.*]]: memref<?xi32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, 
// CHECK-SAME:                           %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:           %[[VAL_10:.*]] = arith.constant 300 : index
// CHECK:           %[[VAL_11:.*]] = arith.constant 128 : index
// CHECK:           %[[VAL_12:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_13:.*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_14:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_16:.*]] = arith.constant 128 : i32
// CHECK:           %[[VAL_17:.*]] = arith.muli %[[VAL_7]], %[[VAL_16]] : i32
// CHECK:           %[[VAL_18:.*]] = arith.index_cast %[[VAL_17]] : i32 to index
// CHECK:           %[[VAL_19:.*]] = scf.for %[[VAL_20:.*]] = %[[VAL_15]] to %[[VAL_13]] step %[[VAL_14]] iter_args(%[[VAL_21:.*]] = %[[VAL_18]]) -> (index)  : i32 {
// CHECK:             %[[VAL_22:.*]] = scf.for %[[VAL_23:.*]] = %[[VAL_15]] to %[[VAL_13]] step %[[VAL_14]] iter_args(%[[VAL_24:.*]] = %[[VAL_21]]) -> (index)  : i32 {
// CHECK:               %[[VAL_25:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_24]]], sizes: [128], strides: {{\[}}%[[VAL_12]]] : memref<?xi32> to memref<128xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_26:.*]] = memref.alloc() : memref<128xi32>
// CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_24]], %[[VAL_11]] : index
// CHECK:               %[[VAL_28:.*]] = arith.maxsi %[[VAL_24]], %[[VAL_10]] : index
// CHECK:               %[[VAL_29:.*]] = arith.minsi %[[VAL_27]], %[[VAL_28]] : index
// CHECK:               %[[VAL_30:.*]] = arith.subi %[[VAL_29]], %[[VAL_24]] : index
// CHECK:               %[[VAL_31:.*]] = arith.cmpi slt, %[[VAL_30]], %[[VAL_11]] : index
// CHECK:               scf.if %[[VAL_31]] {
// CHECK:                 linalg.fill ins(%[[VAL_15]] : i32) outs(%[[VAL_26]] : memref<128xi32>)
// CHECK:               }
// CHECK:               %[[VAL_32:.*]] = memref.subview %[[VAL_25]][0] {{\[}}%[[VAL_30]]] [1] : memref<128xi32, strided<[?], offset: ?>> to memref<?xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_33:.*]] = memref.subview %[[VAL_26]][0] {{\[}}%[[VAL_30]]] [1] : memref<128xi32> to memref<?xi32, strided<[1]>>
// CHECK:               memref.copy %[[VAL_32]], %[[VAL_33]] : memref<?xi32, strided<[?], offset: ?>> to memref<?xi32, strided<[1]>>
// CHECK:               %[[VAL_34:.*]] = bufferization.to_tensor %[[VAL_26]] restrict writable : memref<128xi32>
// CHECK:               %[[VAL_35:.*]] = memref.reinterpret_cast %[[VAL_3]] to offset: {{\[}}%[[VAL_24]]], sizes: [128], strides: {{\[}}%[[VAL_12]]] : memref<?xi32> to memref<128xi32, strided<[?], offset: ?>>
// CHECK:               %[[VAL_36:.*]] = tensor.extract_slice %[[VAL_34]][0] {{\[}}%[[VAL_30]]] [1] : tensor<128xi32> to tensor<?xi32>
// CHECK:               %[[VAL_37:.*]] = memref.subview %[[VAL_35]][0] {{\[}}%[[VAL_30]]] [1] : memref<128xi32, strided<[?], offset: ?>> to memref<?xi32, strided<[?], offset: ?>>
// CHECK:               bufferization.materialize_in_destination %[[VAL_36]] in writable %[[VAL_37]] : (tensor<?xi32>, memref<?xi32, strided<[?], offset: ?>>) -> ()
// CHECK:               %[[VAL_38:.*]] = arith.addi %[[VAL_24]], %[[VAL_11]] : index
// CHECK:               scf.yield %[[VAL_38]] : index
// CHECK:             }
// CHECK:             %[[VAL_39:.*]] = arith.addi %[[VAL_22]], %[[VAL_11]] : index
// CHECK:             scf.yield %[[VAL_39]] : index
// CHECK:           }
// CHECK:           return
// CHECK:         }