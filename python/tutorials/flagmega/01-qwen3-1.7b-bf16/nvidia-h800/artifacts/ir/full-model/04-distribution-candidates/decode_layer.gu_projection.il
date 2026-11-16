// FlagMega high_level stage=distribution_candidates semantic_hash=fb700e96cd24c9d4b2b2a238da93e1e39d27d1bd248b44d51284335c4fdecb74
%decode_layer.gu_projection = fn(%decode_layer.gu_projection.arg0: bf16[2048,12288], %decode_layer.gu_projection.arg1: bf16[1,2048]): // (bf16[2048,12288], bf16[1,2048]) -> (f32[2,12288])
{
  %0 = NTT.SplitKDenseGluProjection(%decode_layer.gu_projection.arg1, %decode_layer.gu_projection.arg0): // f32[2,12288] name='decode_layer_mlp_gate_up.split_projection'
  return (%0)
}
