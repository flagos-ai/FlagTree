// RUN triton-adapter-opt --triton-linearize '--discrete-mask-access-conversion=compile-on-910-95=True force-simt-template=True' --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=True force-simt-template=True' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=True' %s --split-input-file | FileCheck %s

tt.func public @triton_indirect_store_no_repeat_kernel(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i64>, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
  %cst = arith.constant dense<4096> : tensor<4096xi32>
  %0 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
  %1 = tt.splat %arg1 : !tt.ptr<i64> -> tensor<4096x!tt.ptr<i64>>
  %2 = tt.addptr %1, %0 : tensor<4096x!tt.ptr<i64>>, tensor<4096xi32>
  %3 = tt.load %2 : tensor<4096x!tt.ptr<i64>>
  %4 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
  %5 = tt.addptr %4, %3 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi64>
  %6 = tt.load %5 : tensor<4096x!tt.ptr<f32>>
  %7 = arith.cmpi slt, %0, %cst : tensor<4096xi32>
  %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
  %9 = tt.addptr %8, %3 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi64>
  tt.store %9, %6, %7 : tensor<4096x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL:     func.func private @triton_indirect_load(memref<?xf32>, tensor<4096xi64>) -> tensor<4096xf32>
// CHECK-LABEL:     func.func private @triton_indirect_store(memref<?xf32>, tensor<4096xi64>, tensor<4096xf32>, tensor<4096xi1>)
// CHECK-LABEL:     func.func @triton_indirect_store_no_repeat_kernel(
// CHECK-SAME:        %[[VAL_0:.*]]: memref<?xi8>, %[[VAL_1:.*]]: memref<?xi8>, %[[VAL_2:.*]]: memref<?xf32> {tt.tensor_kind = 1 : i32}, %[[VAL_3:.*]]: memref<?xi64> {tt.tensor_kind = 0 : i32}, %[[VAL_4:.*]]: memref<?xf32> {tt.tensor_kind = 0 : i32}, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32, %[[VAL_7:.*]]: i32, %[[VAL_8:.*]]: i32, %[[VAL_9:.*]]: i32, %[[VAL_10:.*]]: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "mix_simd_simt"} {
// CHECK:             %[[VAL_11:.*]] = arith.constant 4096 : i32
// CHECK:             %[[VAL_12:.*]] = tensor.empty() : tensor<4096xi32>
// CHECK:             %[[VAL_13:.*]] = linalg.fill ins(%[[VAL_11:.*]] : i32) outs(%[[VAL_12:.*]] : tensor<4096xi32>) -> tensor<4096xi32>
// CHECK:             %[[VAL_14:.*]] = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%[[VAL_12:.*]] : tensor<4096xi32>) {
// CHECK:             ^bb0(%out: i32):
// CHECK:               %[[VAL_18:.*]] = linalg.index 0 : index
// CHECK:               %[[VAL_19:.*]] = arith.index_cast %[[VAL_18:.*]] : index to i32
// CHECK:               linalg.yield %[[VAL_19:.*]] : i32
// CHECK:             } -> tensor<4096xi32>
// CHECK:             %reinterpret_cast = memref.reinterpret_cast %[[VAL_3:.*]] to offset: [0], sizes: [4096], strides: [1] : memref<?xi64> to memref<4096xi64, strided<[1]>>
// CHECK:             %alloc = memref.alloc() : memref<4096xi64>
// CHECK:             memref.copy %reinterpret_cast, %alloc : memref<4096xi64, strided<[1]>> to memref<4096xi64>
// CHECK:             %[[VAL_15:.*]] = bufferization.to_tensor %alloc restrict writable : memref<4096xi64>
// CHECK:             %[[VAL_16:.*]] = call @triton_indirect_load(%[[VAL_4:.*]], %[[VAL_15:.*]]) : (memref<?xf32>, tensor<4096xi64>) -> tensor<4096xf32>
// CHECK:             %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_14:.*]], %[[VAL_13:.*]] : tensor<4096xi32>
// CHECK:             call @triton_indirect_store(%[[VAL_2:.*]], %[[VAL_15:.*]], %[[VAL_16:.*]], %[[VAL_17:.*]]) : (memref<?xf32>, tensor<4096xi64>, tensor<4096xf32>, tensor<4096xi1>) -> ()
// CHECK:             return
// CHECK:           }

