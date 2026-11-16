// FlagMega high_level stage=unused_functions_removed semantic_hash=3b4f22ecd8aa2a6577a31ac24a9ffc28d6c7ec197bc411c9b6887803a5df1dde
%decode_layer.before_gu = fn(%decode_layer.before_gu.arg0: f32[1,1,1], %decode_layer.before_gu.arg1: f32<4>[1,512], %decode_layer.before_gu.arg2: &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, %decode_layer.before_gu.arg3: bf16<4>[512], %decode_layer.before_gu.arg4: bf16<8,2,8>[128,256], %decode_layer.before_gu.arg5: bf16<8,2,8>[128,128], %decode_layer.before_gu.arg6: bf16<8,2,8>[128,128], %decode_layer.before_gu.arg7: bf16<8>[16], %decode_layer.before_gu.arg8: bf16<8>[16], %decode_layer.before_gu.arg9: bf16<8,2,8>[128,256], %decode_layer.before_gu.arg10: bf16<4>[512], %decode_layer.before_gu.arg11: i32[], %decode_layer.before_gu.arg12: bool[], %decode_layer.before_gu.arg13: bf16<2,8>[1,1,8], %decode_layer.before_gu.arg14: bf16<2,8>[1,1,8]): // (f32[1,1,1], f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, bf16<4>[512], bf16<8,2,8>[128,256], bf16<8,2,8>[128,128], bf16<8,2,8>[128,128], bf16<8>[16], bf16<8>[16], bf16<8,2,8>[128,256], bf16<4>[512], i32[], bool[], bf16<2,8>[1,1,8], bf16<2,8>[1,1,8]) -> (&paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32<4>[1,512], bf16[1,2048])
{
  weights {
    // display grouping only; not an execution schedule
    constant_recipe constant_recipe_0001 // fingerprint=4d29a116f27623a393785339725cfdef6ce70286c6c98bece27ea2eafc54a4e1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], const(bf16[2048] : splat(0.0))): // bf16<4>[512] name='decode_layer_input_norm.vectorized.pack3'
      yield (%0)
    }
    constant_recipe constant_recipe_0002 // fingerprint=70df00e696ae87945c989d1d30cdd31288d6dd82bd4444c60312e738fe4b765c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], const(bf16[128] : splat(0.0))): // bf16<8>[16] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.q_bias'
      yield (%0)
    }
    %w0 = None(): // none name='decode_layer_qkv_none'
  }
  // compute
  %0  = NN.BindNormStats(Axis: 1, UseMean: False, %decode_layer.before_gu.arg1, %decode_layer.before_gu.arg0): // f32[1,1,1] name='decode_layer_input_norm.stats'
  %1  = NN.NormApply(Axis: -1, Epsilon: 1e-06, OutputDtype: 'bfloat16', RoundBeforeScale: False, UseMean: False, %decode_layer.before_gu.arg1, %0, %decode_layer.before_gu.arg3, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0001', Output: 'decode_layer_input_norm.vectorized.pack3')): // bf16<4>[1,512] name='decode_layer_input_norm.vectorized.compute'
  %2  = Tensors.Bitcast(Dtype: 'bfloat16', %1): // bf16[1,2048] name='decode_layer_input_norm'
  %3  = NTT.PackedQKVParallelLinear(NumHeads: 16, NumKvHeads: 8, OutputDataType: 'bfloat16', RhsLayout: 'k_major', %2, %decode_layer.before_gu.arg4, %decode_layer.before_gu.arg5, %decode_layer.before_gu.arg6, %w0, %w0, %w0, %w0, %w0, %w0, %w0, %w0, %w0): // (bf16<8>[1,256], bf16<8>[1,128], bf16<8>[1,128]) name='decode_layer_qkv_projection.packed_projection'
  %4  = NTT.PackedQKVParallelLinearCombine(OutputType: (bf16<8>[1,256], bf16<8>[1,128], bf16<8>[1,128]), %3): // (bf16<8>[1,256], bf16<8>[1,128], bf16<8>[1,128]) name='decode_layer_qkv_projection.packed_combine'
  %5  = GetItem(%4, 0): // bf16<8>[1,256] name='decode_layer_qkv_projection.q.packed'
  %6  = Tensors.Reshape(Shape: [1, 16, 16], %5): // bf16<8>[1,16,16] name='decode_layer_query_norm.stats.vectorized.pack0'
  %7  = GetItem(%4, 1): // bf16<8>[1,128] name='decode_layer_qkv_projection.k.packed'
  %8  = Tensors.Reshape(Shape: [1, 8, 16], %7): // bf16<8>[1,8,16] name='decode_layer_key_norm.stats.vectorized.pack0'
  %9  = GetItem(%4, 2): // bf16<8>[1,128] name='decode_layer_qkv_projection.v.packed'
  %10 = Tensors.Reshape(Shape: [1, 8, 16], %9): // bf16<8>[1,8,16] name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.v'
  %11 = Tuple(%6, %8, %10): // (bf16<8>[1,16,16], bf16<8>[1,8,16], bf16<8>[1,8,16]) name='decode_layer_updated_state.qkv_rope_with_cache.vectorized.qkv'
  %12 = NN.NormStats(Axis: -1, UseMean: False, %6): // f32[1,1,16,1] name='decode_layer_query_norm.stats'
  %13 = NN.NormStats(Axis: -1, UseMean: False, %8): // f32[1,1,8,1] name='decode_layer_key_norm.stats'
  %14 = NN.QKVRoPEWithCache(AttentionLayout: ['seq', 'head', 'dim'], KAxis: -1, KEpsilon: 1e-06, KRoundBeforeScale: False, KUseMean: False, QAxis: -1, QEpsilon: 1e-06, QRoundBeforeScale: False, QUseMean: False, QkvLayout: ['seq', 'head', 'dim'], RoundQkIntermediates: False, %11, %decode_layer.before_gu.arg7, %decode_layer.before_gu.arg8, ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0002', Output: 'decode_layer_updated_state.qkv_rope_with_cache.vectorized.q_bias'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0002', Output: 'decode_layer_updated_state.qkv_rope_with_cache.vectorized.q_bias'), %decode_layer.before_gu.arg13, %decode_layer.before_gu.arg14, %decode_layer.before_gu.arg2, %decode_layer.before_gu.arg11, %decode_layer.before_gu.arg12, %12, %13): // (bf16<8>[1,16,16], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}) !read_write<paged_attention_kv_cache> name='decode_layer_updated_state.qkv_rope_with_cache'
  %15 = GetItem(%14, 0): // bf16<8>[1,16,16] name='decode_layer_updated_state.qkv_rope_with_cache.query'
  %16 = GetItem(%14, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='decode_layer_updated_state'
  %17 = NTT.PagedAttentionPartial(HiddenSize: 2048, Layout: ['seq', 'head', 'dim'], Scale: 0.08838834764831845, SplitCount: 16, SplitHierarchyAxis: 1, %15, %16, %decode_layer.before_gu.arg11): // (f32[1,16,1], f32[1,16,1], f32[1,16,128]) !read<paged_attention_kv_cache> name='decode_layer_paged_attention.partial'
  %18 = GetItem(%17, 0): // f32[1,16,1] name='decode_layer_paged_attention.max_state'
  %19 = GetItem(%17, 1): // f32[1,16,1] name='decode_layer_paged_attention.sum_state'
  %20 = GetItem(%17, 2): // f32[1,16,128] name='decode_layer_paged_attention.acc_state'
  %21 = NTT.PagedAttentionCombine(HiddenSize: 2048, Layout: ['seq', 'head', 'dim'], OutputDataType: {elem_type: 'bfloat16', kind: 'vector', lanes: [8]}, OutputType: bf16<8>[1,16,16], SplitCount: 16, SplitHierarchyAxis: 1, %18, %19, %20): // bf16<8>[1,16,16] name='decode_layer_paged_attention'
  %22 = Tensors.Bitcast(Dtype: {elem_type: 'float32', kind: 'vector', lanes: [2, 2]}, %decode_layer.before_gu.arg1): // f32<2,2>[1,512] name='decode_layer_after_attention.vectorized.pack0.propagated.pack.propagated.pack'
  %23 = Tensors.Reshape(Shape: [1, 256], %21): // bf16<8>[1,256] name='decode_layer_attention_merged.propagated.reshape'
  %24 = Tensors.Bitcast(Dtype: 'bfloat16', %23): // bf16[1,2048] name='decode_layer_attention_merged'
  %25 = NTT.PackedMatMul(FusedReduce: False, OutputDataType: 'bfloat16', RhsLayout: 'k_major', %24, %decode_layer.before_gu.arg9, %w0, %w0): // bf16<8>[1,256] name='decode_layer_attention_output.vectorized.compute'
  %26 = Tensors.Bitcast(Dtype: 'bfloat16', %25): // bf16[1,2048] name='decode_layer_after_attention.projection_wide.cast_input'
  %27 = Math.VectorizedBinary(BinaryOp: 'add', PreOps: {lhs: fusion(value: f32<2,2>[1,512]) => NTT.VectorizedCast(NTT.VectorizedCast(value, NewType: {elem_type: 'bfloat16', kind: 'vector', lanes: [4]}, RoundingContract: 'preserve', VectorizeAxes: [1]), NewType: {elem_type: 'float32', kind: 'vector', lanes: [4]}, RoundingContract: 'preserve', VectorizeAxes: [1]), rhs: fusion(value: bf16[1,2048]) => Tensors.Cast(value, Dtype: 'float32', RoundingContract: 'preserve')}, RhsPack: {axes: [1], lanes: [4]}, %22, %26): // f32<4>[1,512] name='decode_layer_after_attention.vectorized.compute'
  %28 = NN.NormStats(Axis: -1, UseMean: False, %27): // f32[1,1,1] name='decode_layer_post_attention_norm.stats'
  %29 = NN.NormApply(Axis: -1, Epsilon: 1e-06, OutputDtype: 'bfloat16', RoundBeforeScale: False, UseMean: False, %27, %28, %decode_layer.before_gu.arg10, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0001', Output: 'decode_layer_input_norm.vectorized.pack3')): // bf16<4>[1,512] name='decode_layer_post_attention_norm.vectorized.compute'
  %30 = Tensors.Bitcast(Dtype: 'bfloat16', %29): // bf16[1,2048] name='decode_layer_post_attention_norm'
  return (%16, %27, %30)
}

// selections
select vectorization.decode_layer_input_norm = vectorization.norm_apply.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.decode_layer_query_norm.stats = vectorization.norm_stats.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.decode_layer_key_norm.stats = vectorization.norm_stats.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.decode_layer_updated_state.qkv_rope_with_cache = vectorization.qkv_rope_with_cache.cache_layout // kind=vectorization, origin=default-policy
select vectorization.decode_layer_post_attention_norm.stats = vectorization.norm_stats.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.decode_layer_post_attention_norm = vectorization.norm_apply.reduction_axis // kind=vectorization, origin=default-policy
select packing.decode_layer_qkv_projection = packing.k_major_n8_k16 // kind=packing, origin=default-policy
select packing.decode_layer_attention_output = packing.k_major_n8_k16 // kind=packing, origin=default-policy
