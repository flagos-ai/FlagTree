// RUN: triton-shared-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func @kernel(
  %arg0 : !tt.ptr<bf16>,
  %arg1 : !tt.ptr<bf16>,
  %arg2 : i32
  ) {
    %0 = tt.addptr %arg0, %arg2 : !tt.ptr<bf16>, i32
    %1 = tt.addptr %arg1, %arg2 : !tt.ptr<bf16>, i32
    %10 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false}: !tt.ptr<bf16>
    tt.store %1, %10 : !tt.ptr<bf16>
    tt.return
  }
}

// CHECK: module {
// CHECK:   func.func @kernel(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32) {
// CHECK:     %0 = arith.index_cast %arg2 : i32 to index
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%0], sizes: [1], strides: [1] : memref<*xbf16> to memref<1xbf16, strided<[1], offset: ?>>
// CHECK:     %1 = arith.index_cast %arg2 : i32 to index
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%1], sizes: [1], strides: [1] : memref<*xbf16> to memref<1xbf16, strided<[1], offset: ?>>
// CHECK:     %2 = affine.load %reinterpret_cast[0] : memref<1xbf16, strided<[1], offset: ?>>
// CHECK:     affine.store %2, %reinterpret_cast_0[0] : memref<1xbf16, strided<[1], offset: ?>>
// CHECK:     return
// CHECK:   }
// CHECK: }
