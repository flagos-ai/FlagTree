// RUN: triton-shared-opt --triton-ptr-to-memref %s | FileCheck %s

module {
  func.func public @add_kernel_01234(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) {
    %0 = builtin.unrealized_conversion_cast %arg2 : !tt.ptr<f32> to memref<*xf32>
    %1 = builtin.unrealized_conversion_cast %arg1 : !tt.ptr<f32> to memref<*xf32>
    %2 = builtin.unrealized_conversion_cast %arg0 : !tt.ptr<f32> to memref<*xf32>
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %3 = tt.get_program_id x : i32
    %4 = arith.muli %3, %c1024_i32 : i32
    %5 = arith.index_cast %4 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %2 to offset: [%5], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %6 = arith.addi %5, %c1024 : index
    %7 = arith.index_cast %arg3 : i32 to index
    %8 = arith.minsi %6, %7 : index
    %9 = arith.subi %8, %5 : index
    %alloc = memref.alloc() : memref<1024xf32>
    %subview = memref.subview %reinterpret_cast[0] [%9] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%9] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %10 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32> to tensor<1024xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %1 to offset: [%5], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1024xf32>
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%9] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%9] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %11 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32> to tensor<1024xf32>
    %12 = arith.addf %10, %11 : tensor<1024xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %0 to offset: [%5], sizes: [1024], strides: [1] : memref<*xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %12[0] [%9] [1] : tensor<1024xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%9] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}

// CHECK:   func.func public @add_kernel_01234(%arg0: memref<*xf32>, %arg1: memref<*xf32>, %arg2: memref<*xf32>, %arg3: i32)
// CHECK-NOT: builtin.unrealized_conversion_cast %arg2 : memref<*xf32> to !tt.ptr<f32>
// CHECK-NOT: builtin.unrealized_conversion_cast %arg1 : memref<*xf32> to !tt.ptr<f32>
// CHECK-NOT: builtin.unrealized_conversion_cast %arg0 : memref<*xf32> to !tt.ptr<f32>
