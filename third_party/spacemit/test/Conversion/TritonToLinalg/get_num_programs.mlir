// XFAIL: *
// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s

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
    tt.store %7, %8 : tensor<1x!tt.ptr<i32>>
    %9 = tt.addptr %6, %4 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %10 = tt.splat %1 : i32 -> tensor<1xi32>
    tt.store %9, %10 : tensor<1x!tt.ptr<i32>>
    %11 = tt.addptr %6, %5 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %12 = tt.splat %2 : i32 -> tensor<1xi32>
    tt.store %11, %12 : tensor<1x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK: module {
// CHECK:   func.func @num_programs(%arg0: memref<*xi32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
// CHECK:     %c2 = arith.constant 2 : index
// CHECK:     %c1 = arith.constant 1 : index
// CHECK:     %c0 = arith.constant 0 : index
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:     %0 = tensor.empty() : tensor<1xi32>
// CHECK:     %1 = linalg.fill ins(%arg1 : i32) outs(%0 : tensor<1xi32>) -> tensor<1xi32>
// CHECK:     bufferization.materialize_in_destination %1 in writable %reinterpret_cast
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%c1], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:     %2 = tensor.empty() : tensor<1xi32>
// CHECK:     %3 = linalg.fill ins(%arg2 : i32) outs(%2 : tensor<1xi32>) -> tensor<1xi32>
// CHECK:     bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0
// CHECK:     %reinterpret_cast_1 = memref.reinterpret_cast %arg0 to offset: [%c2], sizes: [1], strides: [1] : memref<*xi32> to memref<1xi32, strided<[1], offset: ?>>
// CHECK:     %4 = tensor.empty() : tensor<1xi32>
// CHECK:     %5 = linalg.fill ins(%arg3 : i32) outs(%4 : tensor<1xi32>) -> tensor<1xi32>
// CHECK:     bufferization.materialize_in_destination %5 in writable %reinterpret_cast_1
// CHECK:     return
// CHECK:   }
// CHECK: }
