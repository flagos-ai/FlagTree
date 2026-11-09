// RUN: triton-opt '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true' --triton-to-linalg --split-input-file %s | FileCheck %s

// Tests for LoadBroadcastConverter:
//   Case 1: ptr broadcast src shape matches mask broadcast src shape →
//     rewrite to small load (with mask/other) + broadcast of result.
//   Case 2: ptr broadcast src shape ≠ mask broadcast src shape (e.g. different
//     broadcast axes) → rewrite to small load (no mask/other) + broadcast +
//     arith.select with the full mask and other.

// -----

// Case 2: ptr broadcast [1x64] -> [256x64] (axis 0), mask broadcast [256x1] ->
// [256x64] (axis 1). Different broadcast axes → small load [1x64] without
// mask/other, broadcast to full shape, then arith.select with full mask + other.
module {
  tt.func public @kernel_load_diff_axis_broadcast(%X_ptr: !tt.ptr<f32>, %Out_ptr: !tt.ptr<f32>, %M: i32) attributes {noinline = false} {
    %offsets = arith.constant dense<64> : tensor<256x1xi32>
    %val = arith.constant dense<0.000000e+00> : tensor<256x64xf32>
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %offsetx = tt.get_program_id x : i32
    %offsetx_0 = arith.muli %offsetx, %c256_i32 : i32
    %offsetx_1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %offsetx_2 = tt.splat %offsetx_0 : i32 -> tensor<256xi32>
    %offsetx_3 = arith.addi %offsetx_2, %offsetx_1 : tensor<256xi32>
    %offsety = tt.get_program_id y : i32
    %offsety_4 = arith.muli %offsety, %c64_i32 : i32
    %offsety_5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %offsety_6 = tt.splat %offsety_4 : i32 -> tensor<64xi32>
    %offsety_7 = arith.addi %offsety_6, %offsety_5 : tensor<64xi32>
    %ptr = tt.expand_dims %offsety_7 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %ptr_8 = tt.splat %X_ptr : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ptr_9 = tt.addptr %ptr_8, %ptr : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %mask = tt.splat %M : i32 -> tensor<256xi32>
    %mask_10 = arith.cmpi slt, %offsetx_3, %mask : tensor<256xi32>
    %mask_11 = tt.expand_dims %mask_10 {axis = 1 : i32} : tensor<256xi1> -> tensor<256x1xi1>
    %val_12 = tt.broadcast %ptr_9 : tensor<1x64x!tt.ptr<f32>> -> tensor<256x64x!tt.ptr<f32>>
    %val_13 = tt.broadcast %mask_11 : tensor<256x1xi1> -> tensor<256x64xi1>
    %val_14 = tt.load %val_12, %val_13, %val : tensor<256x64x!tt.ptr<f32>>
    %offsets_15 = tt.expand_dims %offsetx_3 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
    %offsets_16 = arith.muli %offsets_15, %offsets : tensor<256x1xi32>
    %offsets_17 = tt.broadcast %offsets_16 : tensor<256x1xi32> -> tensor<256x64xi32>
    %offsets_18 = tt.broadcast %ptr : tensor<1x64xi32> -> tensor<256x64xi32>
    %offsets_19 = arith.addi %offsets_17, %offsets_18 : tensor<256x64xi32>
    %0 = tt.splat %Out_ptr : !tt.ptr<f32> -> tensor<256x64x!tt.ptr<f32>>
    %1 = tt.addptr %0, %offsets_19 : tensor<256x64x!tt.ptr<f32>>, tensor<256x64xi32>
    tt.store %1, %val_14 : tensor<256x64x!tt.ptr<f32>>
    tt.return
  }
}
// CHECK-LABEL: func.func @kernel_load_diff_axis_broadcast
// CHECK:      tensor.extract_slice %{{.*}}[0, 0] [%{{.*}}, 64] [1, 1] : tensor<256x64xf32> to tensor<?x64xf32>
// CHECK:      tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [%{{.*}}, 64] [1, 1] : tensor<?x64xf32> into tensor<256x64xf32>

// -----

// Case 2 variant: same different-broadcast-axes pattern but the load has no
// 'other' operand. The pass creates a default zero constant and inserts an
// arith.select with it.
module {
  tt.func public @kernel_load_diff_axis_broadcast_no_other(%X_ptr: !tt.ptr<f32>, %Out_ptr: !tt.ptr<f32>, %M: i32) attributes {noinline = false} {
    %offsets = arith.constant dense<64> : tensor<256x1xi32>
    %val = arith.constant dense<0.000000e+00> : tensor<256x64xf32>
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %offsetx = tt.get_program_id x : i32
    %offsetx_0 = arith.muli %offsetx, %c256_i32 : i32
    %offsetx_1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %offsetx_2 = tt.splat %offsetx_0 : i32 -> tensor<256xi32>
    %offsetx_3 = arith.addi %offsetx_2, %offsetx_1 : tensor<256xi32>
    %offsety = tt.get_program_id y : i32
    %offsety_4 = arith.muli %offsety, %c64_i32 : i32
    %offsety_5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %offsety_6 = tt.splat %offsety_4 : i32 -> tensor<64xi32>
    %offsety_7 = arith.addi %offsety_6, %offsety_5 : tensor<64xi32>
    %ptr = tt.expand_dims %offsety_7 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %ptr_8 = tt.splat %X_ptr : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ptr_9 = tt.addptr %ptr_8, %ptr : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %mask = tt.splat %M : i32 -> tensor<256xi32>
    %mask_10 = arith.cmpi slt, %offsetx_3, %mask : tensor<256xi32>
    %mask_11 = tt.expand_dims %mask_10 {axis = 1 : i32} : tensor<256xi1> -> tensor<256x1xi1>
    %val_12 = tt.broadcast %ptr_9 : tensor<1x64x!tt.ptr<f32>> -> tensor<256x64x!tt.ptr<f32>>
    %val_13 = tt.broadcast %mask_11 : tensor<256x1xi1> -> tensor<256x64xi1>
    %val_14 = tt.load %val_12, %val_13, %val : tensor<256x64x!tt.ptr<f32>>
    %offsets_15 = tt.expand_dims %offsetx_3 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
    %offsets_16 = arith.muli %offsets_15, %offsets : tensor<256x1xi32>
    %offsets_17 = tt.broadcast %offsets_16 : tensor<256x1xi32> -> tensor<256x64xi32>
    %offsets_18 = tt.broadcast %ptr : tensor<1x64xi32> -> tensor<256x64xi32>
    %offsets_19 = arith.addi %offsets_17, %offsets_18 : tensor<256x64xi32>
    %0 = tt.splat %Out_ptr : !tt.ptr<f32> -> tensor<256x64x!tt.ptr<f32>>
    %1 = tt.addptr %0, %offsets_19 : tensor<256x64x!tt.ptr<f32>>, tensor<256x64xi32>
    tt.store %1, %val_14 : tensor<256x64x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @kernel_load_diff_axis_broadcast_no_other
// CHECK:      tensor.extract_slice %{{.*}}[0, 0] [%{{.*}}, 64] [1, 1] : tensor<256x64xf32> to tensor<?x64xf32>
// CHECK:      tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [%{{.*}}, 64] [1, 1] : tensor<?x64xf32> into tensor<256x64xf32>

// -----

// Negative: the load ptr is not produced by a tt.broadcast (src shape is
// already [1x64] and the Triton frontend did not insert a broadcast before
// the load). LoadBroadcastConverter does not match; load stays unchanged.
module {
  tt.func public @kernel_load_same_axis_broadcast(%X_ptr: !tt.ptr<f32>, %Out_ptr: !tt.ptr<f32>, %YM: i32) attributes {noinline = false} {
    %offsets = arith.constant dense<64> : tensor<256x1xi32>
    %val = arith.constant dense<0.000000e+00> : tensor<1x64xf32>
    %c64_i32 = arith.constant 64 : i32
    %c256_i32 = arith.constant 256 : i32
    %offsetx = tt.get_program_id x : i32
    %offsetx_0 = arith.muli %offsetx, %c256_i32 : i32
    %offsetx_1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %offsetx_2 = tt.splat %offsetx_0 : i32 -> tensor<256xi32>
    %offsetx_3 = arith.addi %offsetx_2, %offsetx_1 : tensor<256xi32>
    %offsety = tt.get_program_id y : i32
    %offsety_4 = arith.muli %offsety, %c64_i32 : i32
    %offsety_5 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %offsety_6 = tt.splat %offsety_4 : i32 -> tensor<64xi32>
    %offsety_7 = arith.addi %offsety_6, %offsety_5 : tensor<64xi32>
    %ptr = tt.expand_dims %offsety_7 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %ptr_8 = tt.splat %X_ptr : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>>
    %ptr_9 = tt.addptr %ptr_8, %ptr : tensor<1x64x!tt.ptr<f32>>, tensor<1x64xi32>
    %mask = tt.splat %YM : i32 -> tensor<64xi32>
    %mask_10 = arith.cmpi slt, %offsety_7, %mask : tensor<64xi32>
    %mask_11 = tt.expand_dims %mask_10 {axis = 0 : i32} : tensor<64xi1> -> tensor<1x64xi1>
    %val_12 = tt.load %ptr_9, %mask_11, %val : tensor<1x64x!tt.ptr<f32>>
    %offsets_13 = tt.expand_dims %offsetx_3 {axis = 1 : i32} : tensor<256xi32> -> tensor<256x1xi32>
    %offsets_14 = arith.muli %offsets_13, %offsets : tensor<256x1xi32>
    %offsets_15 = tt.broadcast %offsets_14 : tensor<256x1xi32> -> tensor<256x64xi32>
    %offsets_16 = tt.broadcast %ptr : tensor<1x64xi32> -> tensor<256x64xi32>
    %offsets_17 = arith.addi %offsets_15, %offsets_16 : tensor<256x64xi32>
    %0 = tt.splat %Out_ptr : !tt.ptr<f32> -> tensor<256x64x!tt.ptr<f32>>
    %1 = tt.addptr %0, %offsets_17 : tensor<256x64x!tt.ptr<f32>>, tensor<256x64xi32>
    %2 = tt.broadcast %val_12 : tensor<1x64xf32> -> tensor<256x64xf32>
    tt.store %1, %2 : tensor<256x64x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @kernel_load_same_axis_broadcast
// CHECK:      bufferization.to_tensor %{{.*}} restrict writable : memref<1x64xf32> to tensor<1x64xf32>
// CHECK:      linalg.broadcast ins(%{{.*}} : tensor<64xf32>) outs(%{{.*}} : tensor<256x64xf32>) dimensions = [0]
