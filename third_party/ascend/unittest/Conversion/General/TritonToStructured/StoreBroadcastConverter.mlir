// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true' --triton-to-linalg --split-input-file %s | FileCheck %s

// Tests for StoreBroadcastConverter: when store.ptr is defined by a
// tt.broadcast, every pointer along the broadcast axis points to the same
// memory location. The converter rewrites to store only the first index
// along each broadcast axis via tensor.extract_slice, using the broadcast
// source pointer.

// -----

// Case 2: ptr broadcast [1x64] -> [256x64] (axis 0), mask broadcast
// [1x64] -> [256x64] (axis 0). Same broadcast axis → mask's broadcast
// source shape [1x64] matches ptr's broadcast source shape [1x64], so
// the mask src is reused directly (no extract_slice for mask).
module {
  tt.func public @kernel_store_same_axis_broadcast(%X_ptr: !tt.ptr<f32>, %Out_ptr: !tt.ptr<f32>, %YM: i32) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %offsety = tt.get_program_id y : i32
    %offsety_0 = arith.muli %offsety, %c64_i32 : i32
    %offsety_1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %offsety_2 = tt.splat %offsety_0 : i32 -> tensor<64xi32>
    %offsety_3 = arith.addi %offsety_2, %offsety_1 : tensor<64xi32>
    %val = tt.expand_dims %offsety_3 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %val_4 = tt.splat %X_ptr : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %val_5 = tt.addptr %val_4, %val : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %val_6 = tt.load %val_5 : tensor<1x64x!tt.ptr<f32>>
    %val_7 = tt.broadcast %val_6 : tensor<1x64xf32> -> tensor<256x64xf32>
    %ptr = tt.splat %Out_ptr : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ptr_8 = tt.addptr %ptr, %val : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %ptr_9 = tt.broadcast %ptr_8 : tensor<1x64x!tt.ptr<f32>> -> tensor<256x64x!tt.ptr<f32>>
    %mask = tt.splat %YM : i32 -> tensor<64xi32>
    %mask_10 = arith.cmpi slt, %offsety_3, %mask : tensor<64xi32>
    %mask_11 = tt.expand_dims %mask_10 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %0 = tt.broadcast %mask_11 : tensor<1x64xi1> -> tensor<256x64xi1>
    tt.store %ptr_9, %val_7, %0 : tensor<256x64x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: func.func @kernel_store_same_axis_broadcast
// CHECK:      %[[VAL1:.*]] = tensor.extract_slice %{{.*}}[0, 0] [1, %{{.*}}] [1, 1] : tensor<1x64xf32> to tensor<1x?xf32>
// CHECK:      %[[VAL2:.*]]= memref.subview %{{.*}}[0, 0] [1, %{{.*}}] [1, 1] : memref<1x64xf32, strided<[64, 1], offset: ?>> to memref<1x?xf32, strided<[64, 1], offset: ?>>
// CHECK:      bufferization.materialize_in_destination %[[VAL1]] in writable %[[VAL2]]: (tensor<1x?xf32>, memref<1x?xf32, strided<[64, 1], offset: ?>>) -> ()

// -----

// Case 3: ptr broadcast [1x64] -> [256x64] (axis 0), no mask.
// StoreBroadcastConverter still fires: value gets extract_slice to [1x64],
// and the store uses the broadcast source pointer without mask.
module {
  tt.func public @kernel_store_no_mask_broadcast(%X_ptr: !tt.ptr<f32>, %Out_ptr: !tt.ptr<f32>) attributes {noinline = false} {
    %c64_i32 = arith.constant 64 : i32
    %offsety = tt.get_program_id y : i32
    %offsety_0 = arith.muli %offsety, %c64_i32 : i32
    %offsety_1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %offsety_2 = tt.splat %offsety_0 : i32 -> tensor<64xi32>
    %offsety_3 = arith.addi %offsety_2, %offsety_1 : tensor<64xi32>
    %val = tt.expand_dims %offsety_3 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %val_4 = tt.splat %X_ptr : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %val_5 = tt.addptr %val_4, %val : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %val_6 = tt.load %val_5 : tensor<1x64x!tt.ptr<f32>>
    %val_7 = tt.broadcast %val_6 : tensor<1x64xf32> -> tensor<256x64xf32>
    %ptr = tt.splat %Out_ptr : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ptr_8 = tt.addptr %ptr, %val : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %ptr_9 = tt.broadcast %ptr_8 : tensor<1x64x!tt.ptr<f32>> -> tensor<256x64x!tt.ptr<f32>>
    tt.store %ptr_9, %val_7 : tensor<256x64x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: func.func @kernel_store_no_mask_broadcast
// CHECK:      %[[VAL1:.*]]= memref.reinterpret_cast %{{.*}}to offset: [%{{.*}}], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:      %[[VAL2:.*]]= memref.reinterpret_cast %{{.*}}to offset: [%{{.*}}], sizes: [1, 64], strides: [64, 1] : memref<?xf32> to memref<1x64xf32, strided<[64, 1], offset: ?>>
// CHECK:      bufferization.materialize_in_destination %{{.*}} in writable %[[VAL2]]: (tensor<1x64xf32>, memref<1x64xf32, strided<[64, 1], offset: ?>>)


// -----

// The pointer is the canonical form of a non-singleton zero-stride dimension:
// an addptr over the non-zero-stride axis, followed by a one-dimensional
// pointer broadcast. StoreConverter must defer this legal form to the later
// pointer-broadcast lowering instead of rebuilding a lower-rank store.
module {
  tt.func public @zero_stride_store_broadcast(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %offsets_2d = tt.expand_dims %offsets {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %ptr_splat = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ptr_small = tt.addptr %ptr_splat, %offsets_2d : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %ptr = tt.broadcast %ptr_small : tensor<1x64x!tt.ptr<f32>> -> tensor<16x64x!tt.ptr<f32>>
    %zero = arith.constant 0.000000e+00 : f32
    %value = tt.splat %zero : f32 -> tensor<16x64xf32>
    %mask = arith.constant dense<true> : tensor<16x64xi1>
    tt.store %ptr, %value, %mask : tensor<16x64x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @zero_stride_store_broadcast
// CHECK: bufferization.materialize_in_destination %{{.*}} in writable %{{.*}} : (tensor<1x64xf32>, memref<1x64xf32, strided<[64, 1]>>) -> ()
