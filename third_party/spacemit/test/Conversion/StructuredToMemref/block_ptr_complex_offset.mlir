// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @add_ptr_into_make_block_ptr(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>) attributes {noinline = false} {
    %c32768_i64 = arith.constant 32768 : i64
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c512_i64 = arith.constant 512 : i64
    %0 = tt.get_program_id x : i32
    %1 = arith.extsi %0 : i32 to i64
    %2 = arith.muli %1, %c32768_i64 : i64
    %3 = tt.addptr %arg0, %2 : !tt.ptr<bf16>, i64
    %4 = tt.make_tensor_ptr %3, [%c512_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x64xbf16>>
    %5 = tt.addptr %arg1, %2 : !tt.ptr<bf16>, i64
    %6 = tt.make_tensor_ptr %5, [%c512_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<512x64xbf16>>
    %7 = tt.load %4 : !tt.ptr<tensor<512x64xbf16>>
    tt.store %6, %7 : !tt.ptr<tensor<512x64xbf16>>
    tt.return
  }
}

// CHECK: module {
// CHECK:   func.func @add_ptr_into_make_block_ptr(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32) {
// CHECK:     %c32768_i64 = arith.constant 32768 : i64
// CHECK:     %0 = arith.extsi %arg5 : i32 to i64
// CHECK:     %1 = arith.muli %0, %c32768_i64 : i64
// CHECK:     %2 = arith.index_cast %1 : i64 to index
// CHECK:     %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%2], sizes: [512, 64], strides: [64, 1] : memref<*xbf16> to memref<512x64xbf16, strided<[64, 1], offset: ?>>
// CHECK:     %reinterpret_cast_0 = memref.reinterpret_cast %arg1 to offset: [%2], sizes: [512, 64], strides: [64, 1] : memref<*xbf16> to memref<512x64xbf16, strided<[64, 1], offset: ?>>
// CHECK:     %alloc = memref.alloc() : memref<512x64xbf16>
// CHECK:     memref.copy %reinterpret_cast, %alloc : memref<512x64xbf16, strided<[64, 1], offset: ?>> to memref<512x64xbf16>
// CHECK:     %3 = bufferization.to_tensor %alloc restrict writable : memref<512x64xbf16>
// CHECK:     bufferization.materialize_in_destination %3 in writable %reinterpret_cast_0 : (tensor<512x64xbf16>, memref<512x64xbf16, strided<[64, 1], offset: ?>>) -> ()
// CHECK:     return
// CHECK:   }
// CHECK: }
