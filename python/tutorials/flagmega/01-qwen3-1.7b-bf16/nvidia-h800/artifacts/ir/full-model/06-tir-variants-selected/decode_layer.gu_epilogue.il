// FlagMega high_level stage=selected_tir_variants semantic_hash=123424b92c4bac836e8689f7841a9774c245af90ea467270856206f3f938623a
%decode_layer.gu_epilogue = fn(%decode_layer.gu_epilogue.arg0: {f32[2,12288], (B,B), [2,12288], }): // ({f32[2,12288], (B,B), [2,12288], }) -> ({bf16[1,6144], (B,S(C@[0])), [1,256@a], })
{
  %0 = NTT.SplitKDenseGluEpilogue(%decode_layer.gu_epilogue.arg0): // {bf16[1,6144], (B,S(C@[0])), [1,256@a], } name='decode_layer_mlp_gate_up'
  return (%0)
}

// selections
select tir.decode_layer_mlp_gate_up = tir.split_k_dense_glu_epilogue.complete_sum // kind=tir, origin=default-policy
