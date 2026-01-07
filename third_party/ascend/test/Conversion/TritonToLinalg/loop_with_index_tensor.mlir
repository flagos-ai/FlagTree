// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

tt.func public @test_kernel(%arg0: !tt.ptr<i32>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant dense<true> : tensor<4xi1>
  %c4_i32 = arith.constant 4 : i32
  %cst_0 = arith.constant dense<0> : tensor<4xi32>
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %1 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<4x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %3 = scf.for %arg1 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg2 = %cst_0) -> (tensor<4xi32>)  : i32 {
    %4 = tt.splat %arg1 : i32 -> tensor<4xi32>
    %5 = tt.addptr %2, %4 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
    %6 = tt.atomic_rmw add, acq_rel, gpu, %5, %arg2, %cst : (tensor<4x!tt.ptr<i32>>, tensor<4xi32>, tensor<4xi1>) -> tensor<4xi32>
    %7 = arith.addi %arg2, %6 : tensor<4xi32>
    scf.yield %7 : tensor<4xi32>
  }
  tt.return
}

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0) -> (d0)>

// CHECK-LABEL:   func.func @test_kernel(
// CHECK-SAME:                           %[[VAL_0:.*]]: memref<?xi8>, %[[VAL_1:.*]]: memref<?xi8>, %[[VAL_2:.*]]: memref<?xi32> {tt.tensor_kind = 2 : i32},
// CHECK-SAME:                           %[[VAL_3:.*]]: i32, %[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:           %[[VAL_9:.*]] = arith.constant 4 : i32
// CHECK:           %[[VAL_10:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_11:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_12:.*]] = tensor.empty() : tensor<4xi32>
// CHECK:           %[[VAL_13:.*]] = linalg.fill ins(%[[VAL_10]] : i32) outs(%[[VAL_12]] : tensor<4xi32>) -> tensor<4xi32>
// CHECK:           %[[VAL_14:.*]] = scf.for %[[VAL_15:.*]] = %[[VAL_10]] to %[[VAL_9]] step %[[VAL_11]] iter_args(%[[VAL_16:.*]] = %[[VAL_13]]) -> (tensor<4xi32>)  : i32 {
// CHECK:             %[[VAL_17:.*]] = arith.index_cast %[[VAL_15]] : i32 to index
// CHECK:             %[[VAL_18:.*]] = memref.reinterpret_cast %[[VAL_2]] to offset: {{\[}}%[[VAL_17]]], sizes: [4], strides: [1] : memref<?xi32> to memref<4xi32, strided<[1], offset: ?>>
// CHECK:             %[[VAL_19:.*]] = bufferization.to_memref %[[VAL_16]] : memref<4xi32>
// CHECK:             linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_18]], %[[VAL_19]] : memref<4xi32, strided<[1], offset: ?>>, memref<4xi32>) outs(%[[VAL_18]] : memref<4xi32, strided<[1], offset: ?>>) attrs =  {GenericAtomicRMW = "add", MemSemantic = "acq_rel", MemSyncScope = "gpu"} {
// CHECK:             ^bb0(%[[VAL_20:.*]]: i32, %[[VAL_21:.*]]: i32, %[[VAL_22:.*]]: i32):
// CHECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_20]], %[[VAL_21]] : i32
// CHECK:               linalg.yield %[[VAL_23]] : i32
// CHECK:             }
// CHECK:             %[[VAL_24:.*]] = memref.alloc() : memref<4xi32>
// CHECK:             memref.copy %[[VAL_18]], %[[VAL_24]] : memref<4xi32, strided<[1], offset: ?>> to memref<4xi32>
// CHECK:             %[[VAL_25:.*]] = bufferization.to_tensor %[[VAL_24]] restrict writable : memref<4xi32>
// CHECK:             %[[VAL_26:.*]] = linalg.generic {indexing_maps = [#[[$ATTR_0]], #[[$ATTR_0]], #[[$ATTR_0]]], iterator_types = ["parallel"]} ins(%[[VAL_16]], %[[VAL_25]] : tensor<4xi32>, tensor<4xi32>) outs(%[[VAL_16]] : tensor<4xi32>) {
// CHECK:             ^bb0(%[[VAL_27:.*]]: i32, %[[VAL_28:.*]]: i32, %[[VAL_29:.*]]: i32):
// CHECK:               %[[VAL_30:.*]] = arith.addi %[[VAL_27]], %[[VAL_28]] : i32
// CHECK:               linalg.yield %[[VAL_30]] : i32
// CHECK:             } -> tensor<4xi32>
// CHECK:             scf.yield %[[VAL_26]] : tensor<4xi32>
// CHECK:           }
// CHECK:           return
// CHECK:         }