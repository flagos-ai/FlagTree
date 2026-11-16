// FlagMega high_level stage=vectorized semantic_hash=2622f2385bf108124e87228378d58833ab6ab12e3e6985782933ce5020d41506
%decode_layer = fn(%decode_layer_hidden: f32[1,2048], %decode_layer_state: &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, %decode_layer_input_norm_weight: bf16[2048], %decode_layer_q_weight: bf16[2048,2048], %decode_layer_k_weight: bf16[1024,2048], %decode_layer_v_weight: bf16[1024,2048], %decode_layer_q_norm_weight: bf16[128], %decode_layer_k_norm_weight: bf16[128], %decode_layer_o_weight: bf16[2048,2048], %decode_layer_post_norm_weight: bf16[2048], %decode_layer_gate_weight: bf16[6144,2048], %decode_layer_up_weight: bf16[6144,2048], %decode_layer_down_weight: bf16[2048,6144], %decode_layer_layer_id: i32[], %decode_layer_advance_sequence: bool[], %decode_layer_query_rope.table_0.bf16.invariant_parameter: bf16[1,1,128], %decode_layer_query_rope.table_1.bf16.invariant_parameter: bf16[1,1,128]): // (f32[1,2048], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, bf16[2048], bf16[2048,2048], bf16[1024,2048], bf16[1024,2048], bf16[128], bf16[128], bf16[2048,2048], bf16[2048], bf16[6144,2048], bf16[6144,2048], bf16[2048,6144], i32[], bool[], bf16[1,1,128], bf16[1,1,128]) -> (f32[1,2048], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]})
{
  weights {
    // display grouping only; not an execution schedule
    %w0  = Tensors.Pack(Axes: [0], Lanes: [4], %decode_layer_input_norm_weight): // bf16<4>[512] name='decode_layer_input_norm.vectorized.pack2'
    %w1  = Tensors.Pack(Axes: [0], Lanes: [4], const(bf16[2048] : splat(0.0))): // bf16<4>[512] name='decode_layer_input_norm.vectorized.pack3'
    %w2  = Tensors.Permute(Axes: [1, 0], %decode_layer_q_weight): // bf16[2048,2048] name='decode_layer_q_weight_kn'
    %w3  = Tensors.Permute(Axes: [1, 0], %decode_layer_k_weight): // bf16[2048,1024] name='decode_layer_k_weight_kn'
    %w4  = Tensors.Permute(Axes: [1, 0], %decode_layer_v_weight): // bf16[2048,1024] name='decode_layer_v_weight_kn'
    %w5  = None(): // none name='decode_layer_qkv_none'
    %w6  = Tensors.Pack(Axes: [0], Lanes: [8], %decode_layer_q_norm_weight): // bf16<8>[16] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.q_scale'
    %w7  = Tensors.Pack(Axes: [0], Lanes: [8], %decode_layer_k_norm_weight): // bf16<8>[16] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.k_scale'
    %w8  = Tensors.Pack(Axes: [0], Lanes: [8], const(bf16[128] : splat(0.0))): // bf16<8>[16] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.q_bias'
    %w9  = Tensors.Pack(Axes: [0], Lanes: [8], const(bf16[128] : splat(0.0))): // bf16<8>[16] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.k_bias'
    %w10 = Tensors.Pack(Axes: [0], Lanes: [8], %decode_layer_o_weight): // bf16<8>[256,2048] name='decode_layer_attention_output.vectorized.pack1'
    %w11 = Tensors.Pack(Axes: [0], Lanes: [4], %decode_layer_post_norm_weight): // bf16<4>[512] name='decode_layer_post_attention_norm.vectorized.pack2'
    %w12 = Tensors.Pack(Axes: [0], Lanes: [8], %decode_layer_down_weight): // bf16<8>[256,6144] name='decode_layer_mlp_down.vectorized.pack1'
  }
  // compute
  %0  = Tensors.Pack(Axes: [1], Lanes: [4], %decode_layer_hidden): // f32<4>[1,512] name='decode_layer_input_norm.stats.vectorized.pack0'
  %1  = NN.NormStats(Axis: -1, UseMean: False, %0): // f32[1,1,1] name='decode_layer_input_norm.stats'
  %2  = NN.NormApply(Axis: -1, Epsilon: 1e-06, OutputDtype: 'bfloat16', RoundBeforeScale: False, UseMean: False, %0, %1, %w0, %w1): // bf16<4>[1,512] name='decode_layer_input_norm.vectorized.compute'
  %3  = Tensors.Unpack(Axes: [1], %2): // bf16[1,2048] name='decode_layer_input_norm'
  %4  = NN.QKVParallelLinear(NumHeads: 16, NumKvHeads: 8, OutputDataType: 'bfloat16', %3, %w2, %w3, %w4, %w5, %w5, %w5, %w5, %w5, %w5, %w5, %w5, %w5): // (bf16[1,2048], bf16[1,1024], bf16[1,1024]) name='decode_layer_qkv_projection'
  %5  = GetItem(%4, 0): // bf16[1,2048] name='decode_layer_query_flat'
  %6  = GetItem(%4, 1): // bf16[1,1024] name='decode_layer_key_flat'
  %7  = GetItem(%4, 2): // bf16[1,1024] name='decode_layer_value_flat'
  %8  = Tensors.Pack(Axes: [1], Lanes: [8], %5): // bf16<8>[1,256] name='decode_layer_query_norm.stats.vectorized.pack0.propagated.pack'
  %9  = Tensors.Reshape(Shape: [1, 16, 16], %8): // bf16<8>[1,16,16] name='decode_layer_query_norm.stats.vectorized.pack0'
  %10 = NN.NormStats(Axis: -1, UseMean: False, %9): // f32[1,1,16,1] name='decode_layer_query_norm.stats'
  %11 = Tensors.Pack(Axes: [1], Lanes: [8], %6): // bf16<8>[1,128] name='decode_layer_key_norm.stats.vectorized.pack0.propagated.pack'
  %12 = Tensors.Reshape(Shape: [1, 8, 16], %11): // bf16<8>[1,8,16] name='decode_layer_key_norm.stats.vectorized.pack0'
  %13 = NN.NormStats(Axis: -1, UseMean: False, %12): // f32[1,1,8,1] name='decode_layer_key_norm.stats'
  %14 = Tensors.Pack(Axes: [1], Lanes: [8], %5): // bf16<8>[1,256] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.q.propagated.pack'
  %15 = Tensors.Reshape(Shape: [1, 16, 16], %14): // bf16<8>[1,16,16] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.q'
  %16 = Tensors.Pack(Axes: [1], Lanes: [8], %6): // bf16<8>[1,128] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.k.propagated.pack'
  %17 = Tensors.Reshape(Shape: [1, 8, 16], %16): // bf16<8>[1,8,16] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.k'
  %18 = Tensors.Pack(Axes: [1], Lanes: [8], %7): // bf16<8>[1,128] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.v.propagated.pack'
  %19 = Tensors.Reshape(Shape: [1, 8, 16], %18): // bf16<8>[1,8,16] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.v'
  %20 = Tuple(%15, %17, %19): // (bf16<8>[1,16,16], bf16<8>[1,8,16], bf16<8>[1,8,16]) name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.qkv'
  %21 = Tensors.Pack(Axes: [2, 2], Lanes: [2, 8], %decode_layer_query_rope.table_0.bf16.invariant_parameter): // bf16<2,8>[1,1,8] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.cos'
  %22 = Tensors.Pack(Axes: [2, 2], Lanes: [2, 8], %decode_layer_query_rope.table_1.bf16.invariant_parameter): // bf16<2,8>[1,1,8] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.sin'
  %23 = NN.QKVRoPEWithCache(AttentionLayout: ['seq', 'head', 'dim'], KAxis: -1, KEpsilon: 1e-06, KRoundBeforeScale: False, KUseMean: False, QAxis: -1, QEpsilon: 1e-06, QRoundBeforeScale: False, QUseMean: False, QkvLayout: ['seq', 'head', 'dim'], RoundQkIntermediates: False, %20, %w6, %w7, %w8, %w9, %21, %22, %decode_layer_state, %decode_layer_layer_id, %decode_layer_advance_sequence, %10, %13): // (bf16<8>[1,16,16], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}) !read_write<paged_attention_kv_cache> name='decode_layer_updated_state.qkv_rope_with_cache'
  %24 = GetItem(%23, 0): // bf16<8>[1,16,16] name='decode_layer_updated_state.qkv_rope_with_cache.query'
  %25 = GetItem(%23, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='decode_layer_updated_state'
  %26 = NN.PagedAttention(HiddenSize: 2048, Layout: ['seq', 'head', 'dim'], Scale: 0.08838834764831845, %24, %25, %decode_layer_layer_id): // bf16<8>[1,16,16] !read<paged_attention_kv_cache> name='decode_layer_paged_attention'
  %27 = Tensors.Pack(Axes: [1, 1], Lanes: [2, 2], %decode_layer_hidden): // f32<2,2>[1,512] name='decode_layer_after_attention.vectorized.pack0.propagated.pack.propagated.pack'
  %28 = NTT.VectorizedCast(NewType: {elem_type: 'bfloat16', kind: 'vector', lanes: [4]}, RoundingContract: 'preserve', VectorizeAxes: [1], %27): // bf16<4>[1,512] name='decode_layer_after_attention.vectorized.pack0.propagated.pack'
  %29 = NTT.VectorizedCast(NewType: {elem_type: 'float32', kind: 'vector', lanes: [4]}, RoundingContract: 'preserve', VectorizeAxes: [1], %28): // f32<4>[1,512] name='decode_layer_after_attention.vectorized.pack0'
  %30 = Tensors.Reshape(Shape: [1, 256], %26): // bf16<8>[1,256] name='decode_layer_attention_merged.propagated.reshape'
  %31 = Tensors.Unpack(Axes: [1], %30): // bf16[1,2048] name='decode_layer_attention_merged'
  %32 = Math.VectorizedMatMul(LhsAxes: [], OutputAxes: [1], OutputLanes: [8], RhsAxes: [0], TransposeA: False, TransposeB: True, %31, %w10): // bf16<8>[1,256] name='decode_layer_attention_output.vectorized.compute'
  %33 = NTT.VectorizedCast(NewType: {elem_type: 'float32', kind: 'vector', lanes: [2, 4]}, RoundingContract: 'preserve', VectorizeAxes: [1], %32): // f32<2,4>[1,256] name='decode_layer_after_attention.projection_wide.propagated.cast'
  %34 = Tensors.Unpack(Axes: [1, 1], %33): // f32[1,2048] name='decode_layer_after_attention.projection_wide'
  %35 = Math.VectorizedBinary(BinaryOp: 'add', RhsPack: {axes: [1], lanes: [4]}, %29, %34): // f32<4>[1,512] name='decode_layer_after_attention.vectorized.compute'
  %36 = NN.NormStats(Axis: -1, UseMean: False, %35): // f32[1,1,1] name='decode_layer_post_attention_norm.stats'
  %37 = NN.NormApply(Axis: -1, Epsilon: 1e-06, OutputDtype: 'bfloat16', RoundBeforeScale: False, UseMean: False, %35, %36, %w11, %w1): // bf16<4>[1,512] name='decode_layer_post_attention_norm.vectorized.compute'
  %38 = Tensors.Unpack(Axes: [1], %37): // bf16[1,2048] name='decode_layer_post_attention_norm'
  %39 = NN.DenseMatMulGlu(Activation: 'silu', GateClampMax: None, RoundActivation: False, UpClamp: None, %38, %decode_layer_gate_weight, %decode_layer_up_weight): // bf16[1,6144] name='decode_layer_mlp_gate_up'
  %40 = Math.VectorizedMatMul(LhsAxes: [], OutputAxes: [1], OutputLanes: [8], RhsAxes: [0], TransposeA: False, TransposeB: True, %39, %w12): // bf16<8>[1,256] name='decode_layer_mlp_down.vectorized.compute'
  %41 = NTT.VectorizedCast(NewType: {elem_type: 'float32', kind: 'vector', lanes: [2, 4]}, RoundingContract: 'preserve', VectorizeAxes: [1], %40): // f32<2,4>[1,256] name='decode_layer_output.projection_wide.propagated.cast'
  %42 = Tensors.Unpack(Axes: [1, 1], %41): // f32[1,2048] name='decode_layer_output.projection_wide'
  %43 = Math.VectorizedBinary(BinaryOp: 'add', RhsPack: {axes: [1], lanes: [4]}, %35, %42): // f32<4>[1,512] name='decode_layer_output.vectorized.compute'
  %44 = Tensors.Unpack(Axes: [1], %43): // f32[1,2048] name='decode_layer_output'
  return (%44, %25)
}

// selections
select vectorization.decode_layer_input_norm.stats = vectorization.norm_stats.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.decode_layer_input_norm = vectorization.norm_apply.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.decode_layer_query_norm.stats = vectorization.norm_stats.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.decode_layer_key_norm.stats = vectorization.norm_stats.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.decode_layer_updated_state.qkv_rope_with_cache = vectorization.qkv_rope_with_cache.cache_layout // kind=vectorization, origin=default-policy
select vectorization.decode_layer_attention_output = vectorization.matmul.n // kind=vectorization, origin=default-policy
select vectorization.decode_layer_after_attention = vectorization.last_axis // kind=vectorization, origin=default-policy
select vectorization.decode_layer_post_attention_norm.stats = vectorization.norm_stats.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.decode_layer_post_attention_norm = vectorization.norm_apply.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.decode_layer_mlp_down = vectorization.matmul.n // kind=vectorization, origin=default-policy
select vectorization.decode_layer_output = vectorization.last_axis // kind=vectorization, origin=default-policy
