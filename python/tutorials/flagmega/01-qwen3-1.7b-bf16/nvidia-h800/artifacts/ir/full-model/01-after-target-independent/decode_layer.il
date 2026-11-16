// FlagMega high_level stage=call_invariants_hoisted semantic_hash=056c61452faaf9b18f464d68b0b0f64ead73e3dc2ffd36dd9f12bc1d71a01a59
%decode_layer = fn(%decode_layer_hidden: f32[1,2048], %decode_layer_state: &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, %decode_layer_input_norm_weight: bf16[2048], %decode_layer_q_weight: bf16[2048,2048], %decode_layer_k_weight: bf16[1024,2048], %decode_layer_v_weight: bf16[1024,2048], %decode_layer_q_norm_weight: bf16[128], %decode_layer_k_norm_weight: bf16[128], %decode_layer_o_weight: bf16[2048,2048], %decode_layer_post_norm_weight: bf16[2048], %decode_layer_gate_weight: bf16[6144,2048], %decode_layer_up_weight: bf16[6144,2048], %decode_layer_down_weight: bf16[2048,6144], %decode_layer_layer_id: i32[], %decode_layer_advance_sequence: bool[], %decode_layer_query_rope.table_0.bf16.invariant_parameter: bf16[1,1,128], %decode_layer_query_rope.table_1.bf16.invariant_parameter: bf16[1,1,128]): // (f32[1,2048], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, bf16[2048], bf16[2048,2048], bf16[1024,2048], bf16[1024,2048], bf16[128], bf16[128], bf16[2048,2048], bf16[2048], bf16[6144,2048], bf16[6144,2048], bf16[2048,6144], i32[], bool[], bf16[1,1,128], bf16[1,1,128]) -> (f32[1,2048], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]})
{
  weights {
    // display grouping only; not an execution schedule
    %w0 = Tensors.Permute(Axes: [1, 0], %decode_layer_q_weight): // bf16[2048,2048] name='decode_layer_q_weight_kn'
    %w1 = Tensors.Permute(Axes: [1, 0], %decode_layer_k_weight): // bf16[2048,1024] name='decode_layer_k_weight_kn'
    %w2 = Tensors.Permute(Axes: [1, 0], %decode_layer_v_weight): // bf16[2048,1024] name='decode_layer_v_weight_kn'
    %w3 = None(): // none name='decode_layer_qkv_none'
  }
  // compute
  %0  = NN.NormStats(Axis: -1, UseMean: False, %decode_layer_hidden): // f32[1,1,1] name='decode_layer_input_norm.stats'
  %1  = NN.NormApply(Axis: -1, Epsilon: 1e-06, OutputDtype: 'bfloat16', RoundBeforeScale: False, UseMean: False, %decode_layer_hidden, %0, %decode_layer_input_norm_weight, const(bf16[2048] : splat(0.0))): // bf16[1,2048] name='decode_layer_input_norm'
  %2  = NN.QKVParallelLinear(NumHeads: 16, NumKvHeads: 8, OutputDataType: 'bfloat16', %1, %w0, %w1, %w2, %w3, %w3, %w3, %w3, %w3, %w3, %w3, %w3, %w3): // (bf16[1,2048], bf16[1,1024], bf16[1,1024]) name='decode_layer_qkv_projection'
  %3  = GetItem(%2, 0): // bf16[1,2048] name='decode_layer_query_flat'
  %4  = Tensors.Reshape(Shape: [1, 16, 128], %3): // bf16[1,16,128] name='decode_layer_query'
  %5  = GetItem(%2, 1): // bf16[1,1024] name='decode_layer_key_flat'
  %6  = Tensors.Reshape(Shape: [1, 8, 128], %5): // bf16[1,8,128] name='decode_layer_key'
  %7  = GetItem(%2, 2): // bf16[1,1024] name='decode_layer_value_flat'
  %8  = Tensors.Reshape(Shape: [1, 8, 128], %7): // bf16[1,8,128] name='decode_layer_value'
  %9  = NN.NormStats(Axis: -1, UseMean: False, %4): // f32[1,1,16,1] name='decode_layer_query_norm.stats'
  %10 = NN.NormStats(Axis: -1, UseMean: False, %6): // f32[1,1,8,1] name='decode_layer_key_norm.stats'
  %11 = Tuple(%4, %6, %8): // (bf16[1,16,128], bf16[1,8,128], bf16[1,8,128]) name='decode_layer_updated_state.qkv'
  %12 = NN.QKVRoPEWithCache(AttentionLayout: ['seq', 'head', 'dim'], KAxis: -1, KEpsilon: 1e-06, KRoundBeforeScale: False, KUseMean: False, QAxis: -1, QEpsilon: 1e-06, QRoundBeforeScale: False, QUseMean: False, QkvLayout: ['seq', 'head', 'dim'], RoundQkIntermediates: False, %11, %decode_layer_q_norm_weight, %decode_layer_k_norm_weight, const(bf16[128] : splat(0.0)), const(bf16[128] : splat(0.0)), %decode_layer_query_rope.table_0.bf16.invariant_parameter, %decode_layer_query_rope.table_1.bf16.invariant_parameter, %decode_layer_state, %decode_layer_layer_id, %decode_layer_advance_sequence, %9, %10): // (bf16<8>[1,16,16], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}) !read_write<paged_attention_kv_cache> name='decode_layer_updated_state.qkv_rope_with_cache'
  %13 = GetItem(%12, 0): // bf16<8>[1,16,16] name='decode_layer_updated_state.qkv_rope_with_cache.query'
  %14 = GetItem(%12, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='decode_layer_updated_state'
  %15 = NN.PagedAttention(HiddenSize: 2048, Layout: ['seq', 'head', 'dim'], Scale: 0.08838834764831845, %13, %14, %decode_layer_layer_id): // bf16<8>[1,16,16] !read<paged_attention_kv_cache> name='decode_layer_paged_attention'
  %16 = Tensors.Unpack(Axes: [2], %15): // bf16[1,16,128] name='decode_layer_paged_attention_unpacked'
  %17 = Tensors.Reshape(Shape: [1, 2048], %16): // bf16[1,2048] name='decode_layer_attention_merged'
  %18 = Math.MatMul(TransposeA: False, TransposeB: True, %17, %decode_layer_o_weight): // bf16[1,2048] name='decode_layer_attention_output'
  %19 = Tensors.Cast(Dtype: 'bfloat16', RoundingContract: 'preserve', %decode_layer_hidden): // bf16[1,2048] name='decode_layer_after_attention.residual_bf16'
  %20 = Tensors.Cast(Dtype: 'float32', RoundingContract: 'preserve', %19): // f32[1,2048] name='decode_layer_after_attention.residual_wide'
  %21 = Tensors.Cast(Dtype: 'float32', RoundingContract: 'preserve', %18): // f32[1,2048] name='decode_layer_after_attention.projection_wide'
  %22 = Math.Add(%20, %21): // f32[1,2048] name='decode_layer_after_attention'
  %23 = NN.NormStats(Axis: -1, UseMean: False, %22): // f32[1,1,1] name='decode_layer_post_attention_norm.stats'
  %24 = NN.NormApply(Axis: -1, Epsilon: 1e-06, OutputDtype: 'bfloat16', RoundBeforeScale: False, UseMean: False, %22, %23, %decode_layer_post_norm_weight, const(bf16[2048] : splat(0.0))): // bf16[1,2048] name='decode_layer_post_attention_norm'
  %25 = NN.DenseMatMulGlu(Activation: 'silu', GateClampMax: None, RoundActivation: False, UpClamp: None, %24, %decode_layer_gate_weight, %decode_layer_up_weight): // bf16[1,6144] name='decode_layer_mlp_gate_up'
  %26 = Math.MatMul(TransposeA: False, TransposeB: True, %25, %decode_layer_down_weight): // bf16[1,2048] name='decode_layer_mlp_down'
  %27 = Tensors.Cast(Dtype: 'float32', RoundingContract: 'preserve', %26): // f32[1,2048] name='decode_layer_output.projection_wide'
  %28 = Math.Add(%22, %27): // f32[1,2048] name='decode_layer_output'
  return (%28, %14)
}
