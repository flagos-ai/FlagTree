// FlagMega high_level stage=unused_functions_removed semantic_hash=c7cb031533c4394b9fbd57422bf56ed8f9d1b56df866c35fa6f7df653455f2df
%decode_layer.gu_projection = fn(%decode_layer.gu_projection.arg0: bf16[2048,12288], %decode_layer.gu_projection.arg1: bf16[1,2048]): // (bf16[2048,12288], bf16[1,2048]) -> (f32[2,12288])
{
  %0 = NTT.SplitKDenseGluProjection(%decode_layer.gu_projection.arg1, %decode_layer.gu_projection.arg0): // f32[2,12288] name='decode_layer_mlp_gate_up.split_projection'
  return (%0)
}
