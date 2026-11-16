// FlagMega high_level stage=selected_tir_variants semantic_hash=d87b43634d1cfa6a43a6c88f907825e32444840fcdae9f56639906b34de2abd3
%decode_layer.gu_projection = fn(%decode_layer.gu_projection.arg0: {bf16[2048,12288], (B,B), [2048,12288], }, %decode_layer.gu_projection.arg1: {bf16[1,2048], (B,B), [1,2048], }): // ({bf16[2048,12288], (B,B), [2048,12288], }, {bf16[1,2048], (B,B), [1,2048], }) -> ({f32[2,12288], (S(C@[1]),S(C@[0])), [1@x,256@y], })
{
  %0 = NTT.SplitKDenseGluProjection(%decode_layer.gu_projection.arg1, %decode_layer.gu_projection.arg0): // {f32[2,12288], (S(C@[1]),S(C@[0])), [1@x,256@y], } name='decode_layer_mlp_gate_up.split_projection'
  return (%0)
}

// selections
select tir.decode_layer_mlp_gate_up.split_projection = tir.split_k_dense_glu_projection.participant_retire // kind=tir, origin=default-policy
