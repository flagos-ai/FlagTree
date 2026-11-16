// FlagMega high_level stage=attention_gate_fused semantic_hash=f2f8365bda1ab8ce505cbd93e94116130a117a260158bdab97bf29b39c468d11
%decode_layer.gu_epilogue = fn(%decode_layer.gu_epilogue.arg0: {f32[2,12288], (B,B), [2,12288], }): // ({f32[2,12288], (B,B), [2,12288], }) -> ({bf16[1,6144], (B,S(C@[0])), [1,256@a], })
{
  %0 = NTT.SplitKDenseGluEpilogue(%decode_layer.gu_epilogue.arg0): // {bf16[1,6144], (B,S(C@[0])), [1,256@a], } name='decode_layer_mlp_gate_up'
  return (%0)
}
