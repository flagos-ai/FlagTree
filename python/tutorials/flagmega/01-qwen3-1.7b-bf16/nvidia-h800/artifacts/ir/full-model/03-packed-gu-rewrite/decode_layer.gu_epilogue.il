// FlagMega high_level stage=unused_functions_removed semantic_hash=20a078907a6d323fd1328846be300a85aa7b09b55a7eb433e7b05407dd3b39c1
%decode_layer.gu_epilogue = fn(%decode_layer.gu_epilogue.arg0: f32[2,12288]): // (f32[2,12288]) -> (bf16[1,6144])
{
  %0 = NTT.SplitKDenseGluEpilogue(%decode_layer.gu_epilogue.arg0): // bf16[1,6144] name='decode_layer_mlp_gate_up'
  return (%0)
}
