// FlagMega high_level stage=distribution_candidates semantic_hash=72e1a0afc7df1c543dc68ba27de32c90069e7b9ef2e2e1c695e23d749a2cdb85
%decode_layer.gu_epilogue = fn(%decode_layer.gu_epilogue.arg0: f32[2,12288]): // (f32[2,12288]) -> (bf16[1,6144])
{
  %0 = NTT.SplitKDenseGluEpilogue(%decode_layer.gu_epilogue.arg0): // bf16[1,6144] name='decode_layer_mlp_gate_up'
  return (%0)
}
