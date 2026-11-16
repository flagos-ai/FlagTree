// FlagMega high_level stage=attention_gate_fused semantic_hash=9e29d29d052d268d0c7b4410937d95a870173ec151c0a8b5381dbb8c0a3e84a6
%decode_layer.gu_projection = fn(%decode_layer.gu_projection.arg0: {bf16[2048,12288], (B,B), [2048,12288], }, %decode_layer.gu_projection.arg1: {bf16[1,2048], (B,B), [1,2048], }): // ({bf16[2048,12288], (B,B), [2048,12288], }, {bf16[1,2048], (B,B), [1,2048], }) -> ({f32[2,12288], (S(C@[1]),S(C@[0])), [1@x,256@y], })
{
  %0 = NTT.SplitKDenseGluProjection(%decode_layer.gu_projection.arg1, %decode_layer.gu_projection.arg0): // {f32[2,12288], (S(C@[1]),S(C@[0])), [1@x,256@y], } name='decode_layer_mlp_gate_up.split_projection'
  return (%0)
}
