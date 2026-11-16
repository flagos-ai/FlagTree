// FlagMega high_level stage=unused_functions_removed semantic_hash=8d031338f5afb80f70c4485d00f3d7a9cd54768ad9a3b91274e9697a2c7856ae
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
