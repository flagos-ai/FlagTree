// RUN: triton-opt "--triton-to-linalg=global-kernel=false named-ops=True" --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @_triton_mrope_forward
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[ALLOC_5:.*]] = memref.alloc() : memref<64xf32>
// CHECK: linalg.fill ins(%[[CST]] : f32) outs(%[[ALLOC_5]] : memref<64xf32>)
// CHECK: bufferization.to_tensor %[[ALLOC_5]] restrict writable : memref<64xf32> to tensor<64xf32>

module attributes {hacc.target = #hacc.target<"Ascend910_9382">} {
  tt.func public @_triton_mrope_forward(%q_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %k_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %cos: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %sin: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %num_tokens: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<16x64xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64xf32>
    %cst_1 = arith.constant dense<64> : tensor<16x64xi32>
    %cst_2 = arith.constant dense<64> : tensor<1x64xi32>
    %cst_3 = arith.constant dense<16> : tensor<16x1xi32>
    %cst_4 = arith.constant dense<128> : tensor<16x1xi32>
    %cst_5 = arith.constant dense<64> : tensor<64xi32>
    %cst_6 = arith.constant dense<32> : tensor<64xi32>
    %c64_i32 = arith.constant 64 : i32
    %c2048_i32 = arith.constant 2048 : i32
    %pid = tt.get_program_id x : i32
    %q_ptr_7 = arith.muli %pid, %c2048_i32 : i32
    %q_ptr_8 = tt.addptr %q_ptr, %q_ptr_7 : !tt.ptr<f32>, i32
    %k_ptr_9 = tt.addptr %k_ptr, %q_ptr_7 : !tt.ptr<f32>, i32
    %t_cos = arith.muli %pid, %c64_i32 : i32
    %t_cos_10 = tt.addptr %cos, %t_cos : !tt.ptr<f32>, i32
    %h_cos = arith.muli %num_tokens, %c64_i32 : i32
    %h_cos_11 = tt.addptr %t_cos_10, %h_cos : !tt.ptr<f32>, i32
    %w_cos = tt.addptr %h_cos_11, %h_cos : !tt.ptr<f32>, i32
    %t_sin = tt.addptr %sin, %t_cos : !tt.ptr<f32>, i32
    %h_sin = tt.addptr %t_sin, %h_cos : !tt.ptr<f32>, i32
    %w_sin = tt.addptr %h_sin, %h_cos : !tt.ptr<f32>, i32
    %cos_offsets = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %t_mask = arith.cmpi slt, %cos_offsets, %cst_6 : tensor<64xi32>
    %h_mask = arith.cmpi sge, %cos_offsets, %cst_6 : tensor<64xi32>
    %h_mask_12 = arith.cmpi slt, %cos_offsets, %cst_5 : tensor<64xi32>
    %h_mask_13 = arith.andi %h_mask, %h_mask_12 : tensor<64xi1>
    %w_mask = arith.cmpi sge, %cos_offsets, %cst_5 : tensor<64xi32>
    %w_mask_14 = arith.andi %w_mask, %h_mask_12 : tensor<64xi1>
    %t_cos_row = tt.splat %t_cos_10 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %t_cos_row_15 = tt.addptr %t_cos_row, %cos_offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %t_cos_row_16 = tt.load %t_cos_row_15, %t_mask, %cst_0 : tensor<64x!tt.ptr<f32>>
    %h_cos_row = tt.splat %h_cos_11 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %h_cos_row_17 = tt.addptr %h_cos_row, %cos_offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %h_cos_row_18 = tt.load %h_cos_row_17, %h_mask_13, %cst_0 : tensor<64x!tt.ptr<f32>>
    %w_cos_row = tt.splat %w_cos : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %w_cos_row_19 = tt.addptr %w_cos_row, %cos_offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %w_cos_row_20 = tt.load %w_cos_row_19, %w_mask_14, %cst_0 : tensor<64x!tt.ptr<f32>>
    %t_sin_row = tt.splat %t_sin : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %t_sin_row_21 = tt.addptr %t_sin_row, %cos_offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %t_sin_row_22 = tt.load %t_sin_row_21, %t_mask, %cst_0 : tensor<64x!tt.ptr<f32>>
    %h_sin_row = tt.splat %h_sin : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %h_sin_row_23 = tt.addptr %h_sin_row, %cos_offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %h_sin_row_24 = tt.load %h_sin_row_23, %h_mask_13, %cst_0 : tensor<64x!tt.ptr<f32>>
    %w_sin_row = tt.splat %w_sin : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %w_sin_row_25 = tt.addptr %w_sin_row, %cos_offsets : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %w_sin_row_26 = tt.load %w_sin_row_25, %w_mask_14, %cst_0 : tensor<64x!tt.ptr<f32>>
    %cos_row = arith.addf %t_cos_row_16, %h_cos_row_18 : tensor<64xf32>
    %cos_row_27 = arith.addf %cos_row, %w_cos_row_20 : tensor<64xf32>
    %sin_row = arith.addf %t_sin_row_22, %h_sin_row_24 : tensor<64xf32>
    %sin_row_28 = arith.addf %sin_row, %w_sin_row_26 : tensor<64xf32>
    %first_half_q_offsets = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %first_half_q_offsets_29 = tt.expand_dims %first_half_q_offsets {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %first_half_q_offsets_30 = arith.muli %first_half_q_offsets_29, %cst_4 : tensor<16x1xi32>
    %first_half_q_offsets_31 = tt.expand_dims %cos_offsets {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
    %first_half_q_offsets_32 = tt.broadcast %first_half_q_offsets_30 : tensor<16x1xi32> -> tensor<16x64xi32>
    %first_half_q_offsets_33 = tt.broadcast %first_half_q_offsets_31 : tensor<1x64xi32> -> tensor<16x64xi32>
    %first_half_q_offsets_34 = arith.addi %first_half_q_offsets_32, %first_half_q_offsets_33 : tensor<16x64xi32>
    %first_q_mask = arith.cmpi slt, %first_half_q_offsets_29, %cst_3 : tensor<16x1xi32>
    %first_q_mask_35 = arith.cmpi slt, %first_half_q_offsets_31, %cst_2 : tensor<1x64xi32>
    %first_q_mask_36 = tt.broadcast %first_q_mask : tensor<16x1xi1> -> tensor<16x64xi1>
    %first_q_mask_37 = tt.broadcast %first_q_mask_35 : tensor<1x64xi1> -> tensor<16x64xi1>
    %first_q_mask_38 = arith.andi %first_q_mask_36, %first_q_mask_37 : tensor<16x64xi1>
    %q_tile_1 = tt.splat %q_ptr_8 : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
    %q_tile_1_39 = tt.addptr %q_tile_1, %first_half_q_offsets_34 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    %q_tile_1_40 = tt.load %q_tile_1_39, %first_q_mask_38, %cst : tensor<16x64x!tt.ptr<f32>>
    %k_tile_1 = tt.splat %k_ptr_9 : !tt.ptr<f32> -> tensor<16x64x!tt.ptr<f32>>
    %k_tile_1_41 = tt.addptr %k_tile_1, %first_half_q_offsets_34 : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    %k_tile_1_42 = tt.load %k_tile_1_41, %first_q_mask_38, %cst : tensor<16x64x!tt.ptr<f32>>
    %second_half_q_offsets = arith.addi %first_half_q_offsets_34, %cst_1 : tensor<16x64xi32>
    %q_tile_2 = tt.addptr %q_tile_1, %second_half_q_offsets : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    %q_tile_2_43 = tt.load %q_tile_2, %first_q_mask_38, %cst : tensor<16x64x!tt.ptr<f32>>
    %k_tile_2 = tt.addptr %k_tile_1, %second_half_q_offsets : tensor<16x64x!tt.ptr<f32>>, tensor<16x64xi32>
    %k_tile_2_44 = tt.load %k_tile_2, %first_q_mask_38, %cst : tensor<16x64x!tt.ptr<f32>>
    %new_q_tile_1 = tt.expand_dims %cos_row_27 {axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
    %new_q_tile_1_45 = tt.broadcast %new_q_tile_1 : tensor<1x64xf32> -> tensor<16x64xf32>
    %new_q_tile_1_46 = arith.mulf %q_tile_1_40, %new_q_tile_1_45 : tensor<16x64xf32>
    %new_q_tile_1_47 = tt.expand_dims %sin_row_28 {axis = 0 : i32} : tensor<64xf32> -> tensor<1x64xf32>
    %new_q_tile_1_48 = tt.broadcast %new_q_tile_1_47 : tensor<1x64xf32> -> tensor<16x64xf32>
    %new_q_tile_1_49 = arith.mulf %q_tile_2_43, %new_q_tile_1_48 : tensor<16x64xf32>
    %new_q_tile_1_50 = arith.subf %new_q_tile_1_46, %new_q_tile_1_49 : tensor<16x64xf32>
    tt.store %q_tile_1_39, %new_q_tile_1_50, %first_q_mask_38 : tensor<16x64x!tt.ptr<f32>>
    %new_q_tile_2 = arith.mulf %q_tile_2_43, %new_q_tile_1_45 : tensor<16x64xf32>
    %new_q_tile_2_51 = arith.mulf %q_tile_1_40, %new_q_tile_1_48 : tensor<16x64xf32>
    %new_q_tile_2_52 = arith.addf %new_q_tile_2, %new_q_tile_2_51 : tensor<16x64xf32>
    tt.store %q_tile_2, %new_q_tile_2_52, %first_q_mask_38 : tensor<16x64x!tt.ptr<f32>>
    %new_k_tile_1 = arith.mulf %k_tile_1_42, %new_q_tile_1_45 : tensor<16x64xf32>
    %new_k_tile_1_53 = arith.mulf %k_tile_2_44, %new_q_tile_1_48 : tensor<16x64xf32>
    %new_k_tile_1_54 = arith.subf %new_k_tile_1, %new_k_tile_1_53 : tensor<16x64xf32>
    tt.store %k_tile_1_41, %new_k_tile_1_54, %first_q_mask_38 : tensor<16x64x!tt.ptr<f32>>
    %new_k_tile_2 = arith.mulf %k_tile_2_44, %new_q_tile_1_45 : tensor<16x64xf32>
    %new_k_tile_2_55 = arith.mulf %k_tile_1_42, %new_q_tile_1_48 : tensor<16x64xf32>
    %new_k_tile_2_56 = arith.addf %new_k_tile_2, %new_k_tile_2_55 : tensor<16x64xf32>
    tt.store %k_tile_2, %new_k_tile_2_56, %first_q_mask_38 : tensor<16x64x!tt.ptr<f32>>
    tt.return
  }
}
