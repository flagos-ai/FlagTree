// FlagMega high_level stage=distribution_candidates semantic_hash=30ceb7556aab021a6e9bcf1fd94d84739e5216ca9671985cb6dc911280db30c7
%decode_layer.after_gu = fn(%decode_layer.after_gu.arg0: bf16<8,2,8>[384,256], %decode_layer.after_gu.arg1: f32<4>[1,512], %decode_layer.after_gu.arg2: bf16[1,6144]): // (bf16<8,2,8>[384,256], f32<4>[1,512], bf16[1,6144]) -> (f32<4>[1,512], f32[1,1,1])
{
  weights {
    // display grouping only; not an execution schedule
    %w0 = None(): // none name='decode_layer_qkv_none'
  }
  // compute
  %0 = NTT.PackedMatMul(FusedReduce: False, OutputDataType: 'bfloat16', RhsLayout: 'k_major', %decode_layer.after_gu.arg2, %decode_layer.after_gu.arg0, %w0, %w0): // bf16<8>[1,256] name='decode_layer_mlp_down.vectorized.compute'
  %1 = Tensors.Bitcast(Dtype: 'bfloat16', %0): // bf16[1,2048] name='decode_layer_output.projection_wide.cast_input'
  %2 = Math.VectorizedBinary(BinaryOp: 'add', PreOps: {rhs: fusion(value: bf16[1,2048]) => Tensors.Cast(value, Dtype: 'float32', RoundingContract: 'preserve')}, RhsPack: {axes: [1], lanes: [4]}, %decode_layer.after_gu.arg1, %1): // f32<4>[1,512] name='decode_layer_output.vectorized.compute'
  %3 = NN.NormStats(Axis: 1, UseMean: False, %2): // f32[1,1,1] name='decode_layer.output_norm_stats'
  return (%2, %3)
}

// selections
select packing.decode_layer_mlp_down = packing.k_major_n8_k16 // kind=packing, origin=default-policy
select distribution.decode_layer_mlp_down.vectorized.compute = distribution.decode_layer_mlp_down.vectorized.compute.packed_matmul_operand_sbp.mesh_yx_8x16_bb.in_d_b_b__d_b_s_c_h0_1_g2__nonetype__nonetype.out_d_b_s_c_h0_1_g2 // kind=distribution, origin=ortools-cp-sat
select distribution.decode_layer_output.projection_wide.cast_input = distribution.decode_layer_output.projection_wide.cast_input.operation_type_inference_sbp.mesh_yx_8x16_bb.in_d_b_s_c_h0_1_g2.out_d_b_s_c_h0_1_g16 // kind=distribution, origin=ortools-cp-sat
select distribution.decode_layer_output.vectorized.compute = distribution.decode_layer_output.vectorized.compute.operation_type_inference_sbp.mesh_yx_8x16_bb.in_d_b_s_c_h0_1_g4__d_b_s_c_h0_1_g16.out_d_b_s_c_h0_1_g4 // kind=distribution, origin=ortools-cp-sat
select distribution.decode_layer.output_norm_stats = distribution.decode_layer.output_norm_stats.norm_stats_additive_sbp.mesh_yx_8x16_bb.in_d_b_b.out_d_b_b_b // kind=distribution, origin=ortools-cp-sat
