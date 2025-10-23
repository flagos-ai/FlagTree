// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s 
module {
  tt.func public @num_programs(%arg0: !tt.ptr<i32>) {
    %0 = tt.get_num_programs x : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.get_num_programs z : i32
    %3 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
    %4 = tt.make_range {end = 2 : i32, start = 1 : i32} : tensor<1xi32>
    %5 = tt.make_range {end = 3 : i32, start = 2 : i32} : tensor<1xi32>
    %6 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %7 = tt.addptr %6, %3 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %8 = tt.splat %0 : i32 -> tensor<1xi32>
    tt.store %7, %8 evictionPolicy = evict_last : tensor<1x!tt.ptr<i32>>
    %9 = tt.addptr %6, %4 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %10 = tt.splat %1 : i32 -> tensor<1xi32>
    tt.store %9, %10 evictionPolicy = evict_last : tensor<1x!tt.ptr<i32>>
    %11 = tt.addptr %6, %5 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %12 = tt.splat %2 : i32 -> tensor<1xi32>
    tt.store %11, %12 evictionPolicy = evict_last : tensor<1x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: module {
// CHECK:       func.func @num_programs(
// CHECK-SAME:   %arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xi32> {tt.tensor_kind = 1 : i32}, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "", mix_mode = "aiv"} {
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1]>>
// CHECK:     %[[COMMON_EMPTY_TENSOR:.*]] = tensor.empty() : tensor<1xi32>
// CHECK:     %1 = linalg.fill ins(%arg3 : i32) outs(%[[COMMON_EMPTY_TENSOR]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:     bufferization.materialize_in_destination %1 in writable %reinterpret_cast : (tensor<1xi32>, memref<1xi32, strided<[1]>>) -> ()
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast %arg2 to offset: [1], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: 1>>
// CHECK:     %2 = linalg.fill ins(%arg4 : i32) outs(%[[COMMON_EMPTY_TENSOR]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:     bufferization.materialize_in_destination %2 in writable %reinterpret_cast_0 : (tensor<1xi32>, memref<1xi32, strided<[1], offset: 1>>) -> ()
// CHECK:     %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [2], sizes: [1], strides: [1] : memref<?xi32> to memref<1xi32, strided<[1], offset: 2>>
// CHECK:     %3 = linalg.fill ins(%arg5 : i32) outs(%[[COMMON_EMPTY_TENSOR]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK:     bufferization.materialize_in_destination %3 in writable %reinterpret_cast_1
// CHECK:     return
// CHECK:   }
// CHECK: }
