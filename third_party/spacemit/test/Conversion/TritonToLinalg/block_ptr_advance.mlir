// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s

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

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: module {
// CHECK:   func.func @matmul_kernel_with_block_pointers_01234567891011(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32) {
// CHECK:     %c64 = arith.constant 64 : index
// CHECK:     %c256_i32 = arith.constant 256 : i32
// CHECK:     %c0_i32 = arith.constant 0 : i32
// CHECK:     %c64_i32 = arith.constant 64 : i32
// CHECK:     %cst = arith.constant 0.000000e+00 : bf16
// CHECK:     %0 = tensor.empty() : tensor<128x64xbf16>
// CHECK:     %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<128x64xbf16>) -> tensor<128x64xbf16>
// CHECK:     %2 = arith.index_cast %arg12 : i32 to index
// CHECK:     %3 = arith.index_cast %arg6 : i32 to index
// CHECK:     %4 = arith.index_cast %arg7 : i32 to index
// CHECK:     %5 = arith.muli %2, %3 : index
// CHECK:     %6 = arith.muli %4, %c64 : index
// CHECK:     %7 = arith.addi %5, %6 : index
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%7], sizes: [128, 64], strides: [%3, %4] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast %arg0 to offset: [%5], sizes: [128, 64], strides: [%3, %4] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:     %8:5 = scf.for %arg20 = %c0_i32 to %arg5 step %c64_i32 iter_args(%arg21 = %1, %arg22 = %reinterpret_cast, %arg23 = %reinterpret_cast_0, %arg24 = %7, %arg25 = %5) -> (tensor<128x64xbf16>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, index, index)  : i32 {
// CHECK:       %alloc = memref.alloc() : memref<128x64xbf16>
// CHECK:       memref.copy %arg22, %alloc : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<128x64xbf16>
// CHECK:       %17 = bufferization.to_tensor %alloc restrict writable : memref<128x64xbf16>
// CHECK:       %alloc_2 = memref.alloc() : memref<128x64xbf16>
// CHECK:       memref.copy %arg23, %alloc_2 : memref<128x64xbf16, strided<[?, ?], offset: ?>> to memref<128x64xbf16>
// CHECK:       %18 = bufferization.to_tensor %alloc_2 restrict writable : memref<128x64xbf16>
// CHECK:       %19 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%17, %18 : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%17 : tensor<128x64xbf16>) {
// CHECK:       ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
// CHECK:         %25 = arith.addf %in, %in_5 : bf16
// CHECK:         linalg.yield %25 : bf16
// CHECK:       } -> tensor<128x64xbf16>
// CHECK:       %20 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg21, %19 : tensor<128x64xbf16>, tensor<128x64xbf16>) outs(%arg21 : tensor<128x64xbf16>) {
// CHECK:       ^bb0(%in: bf16, %in_5: bf16, %out: bf16):
// CHECK:         %25 = arith.addf %in, %in_5 : bf16
// CHECK:         linalg.yield %25 : bf16
// CHECK:       } -> tensor<128x64xbf16>
// CHECK:       %21 = arith.muli %4, %c64 : index
// CHECK:       %22 = arith.addi %arg24, %21 : index
// CHECK:       %reinterpret_cast_3 = memref.reinterpret_cast %arg0 to offset: [%22], sizes: [128, 64], strides: [%3, %4] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:       %23 = arith.muli %3, %c64 : index
// CHECK:       %24 = arith.addi %23, %arg25 : index
// CHECK:       %reinterpret_cast_4 = memref.reinterpret_cast %arg0 to offset: [%24], sizes: [128, 64], strides: [%3, %4] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:       scf.yield %20, %reinterpret_cast_3, %reinterpret_cast_4, %22, %24 : tensor<128x64xbf16>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, memref<128x64xbf16, strided<[?, ?], offset: ?>>, index, index
// CHECK:     }
// CHECK:     %9 = arith.muli %arg13, %c256_i32 : i32
// CHECK:     %10 = arith.index_cast %arg12 : i32 to index
// CHECK:     %11 = arith.index_cast %9 : i32 to index
// CHECK:     %12 = arith.index_cast %arg10 : i32 to index
// CHECK:     %13 = arith.index_cast %arg11 : i32 to index
// CHECK:     %14 = arith.muli %10, %12 : index
// CHECK:     %15 = arith.muli %11, %13 : index
// CHECK:     %16 = arith.addi %14, %15 : index
// CHECK:     %reinterpret_cast_1 = memref.reinterpret_cast %arg2 to offset: [%16], sizes: [128, 64], strides: [%12, %13] : memref<*xbf16> to memref<128x64xbf16, strided<[?, ?], offset: ?>>
// CHECK:     bufferization.materialize_in_destination %8#0 in writable %reinterpret_cast_1
// CHECK:     return
// CHECK:   }
// CHECK: }
