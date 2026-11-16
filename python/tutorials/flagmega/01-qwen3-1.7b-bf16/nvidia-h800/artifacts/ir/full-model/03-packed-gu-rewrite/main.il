// FlagMega high_level stage=unused_functions_removed semantic_hash=f8f67a7e71633ed6d7a5a9b0eb096fed5135a11adf15593efbd7640fd51f5e5e
%main = fn(%input_ids: i32[1], %paged_attention_kv_cache: &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}): // (i32[1], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}) -> (f32[1,151936], i32[1], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]})
{
  weights {
    // display grouping only; not an execution schedule
    constant_recipe constant_recipe_0000 // fingerprint=db4010db3e79898a8916279a0279faa35a8f54bf53320f099d5e2ef23bc431bf
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [4], WeightRef(bf16[151936,2048], 'embed_tokens.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.embed_tokens.weight')): // bf16<4>[151936,512] name='token_embedding.packed_weight'
      %1 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[151936,2048], 'embed_tokens.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.embed_tokens.weight')): // bf16<8>[151936,256] name='logits.vectorized.compute.rhs_k_major.k_vector'
      %2 = Tensors.Pack(Axes: [1], Lanes: [2], %1): // bf16<2,8>[151936,128] name='logits.vectorized.compute.rhs_k_major.k_pack'
      %3 = Tensors.Pack(Axes: [0], Lanes: [8], %2): // bf16<8,2,8>[18992,128] name='logits.vectorized.compute.rhs_k_major.n_vector'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,18992] name='logits.vectorized.compute.rhs_k_major.physical'
      yield (%0, %4)
    }
    constant_recipe constant_recipe_0001 // fingerprint=4d29a116f27623a393785339725cfdef6ce70286c6c98bece27ea2eafc54a4e1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], const(bf16[2048] : splat(0.0))): // bf16<4>[512] name='decode_layer_input_norm.vectorized.pack3'
      yield (%0)
    }
    constant_recipe constant_recipe_0003 // fingerprint=cd8676be7a012a08eb5fb85b2aba98eac7ed86bb0d9be2832ef68743611ab6d1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.0.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.input_layernorm.weight')): // bf16<4>[512] name='layer_0_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0004 // fingerprint=62ad7a07c82646b38e7c8da9b7e1022a19fffc46c17568079b73a6ccdc82d7c1
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.0.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0005 // fingerprint=0d050d69db060df96381ed4bf34a40a259d2247b1da9d9757f93b251ad074513
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.0.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0006 // fingerprint=3ec0b9f16ae37e37cbc1820685f93279250d0ad4c5dbbbf9970a07eb8a137445
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.0.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0007 // fingerprint=772effe3d4545ee952e823295cbe80c08c60cb0649c02a1232e8bbbb48c05ff2
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.0.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_0_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0008 // fingerprint=4a67d730026ef70f423521034e5fd9930dea25b9a4831e67ef880d32621a28c1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.0.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_0_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0009 // fingerprint=cae0ebefd3484e32697fa23b5bf491c1090fe4cdd67bad122e04df85c6bc7e4a
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.0.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0010 // fingerprint=f3fb71a7c788ab863e49a1621aae085700f13d5419d5c0cc24df4b3aad07989e
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.0.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_0_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0011 // fingerprint=3f6e7086e54256418f10e6ef4817757e4db30e10d39883221877fe6d8422eda3
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.0.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.0.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_0_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_0_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0012 // fingerprint=e6cf5cc51b4625a06535be57880a80b3f7f2c860b2d4bff1581c44212865438f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.0.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0013 // fingerprint=221215640d54bb54ff768043f7268d7a37c3ee479a204a72e1d8e342b6cbd1e9
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.1.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.input_layernorm.weight')): // bf16<4>[512] name='layer_1_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0014 // fingerprint=2a7f61507a24eee99dfb4d951ae427339eb772f000382ff4aa12fd8a2ab7611c
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.1.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0015 // fingerprint=a0166afab52f370ac8561eccd7862a4a792b4486fcea924c04dfe368d00ef41f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.1.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0016 // fingerprint=6d4089328374179e2ff34431d8fbe65b8e0d8058ffdd339d0cecb28fb4ac5a93
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.1.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0017 // fingerprint=286536330e76343e48fcafbb566eed494f0922897ee3ed628a367452644b7e76
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.1.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_1_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0018 // fingerprint=cb9547d2094afdd5787d1508d6dfe949a4f46a38c43233aef605bd056c16374c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.1.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_1_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0019 // fingerprint=c39b5c5b618b4e30acc1232131e6f87c10eab40c8c3eace31b85f080d09d547a
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.1.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0020 // fingerprint=27d779f819f7f4542bbdf32ba0597680ced93a4d3296f5f8d6a6b405f304fd6f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.1.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_1_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0021 // fingerprint=53e07ee94bcb5afe8f2c056240bf386cc7cf33acd1d7c0603699aa36b9082c54
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.1.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.1.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_1_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_1_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0022 // fingerprint=e0cce4da2d52f2d6af036f08285e8888f870758062612e4798ffb64d207dcc21
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.1.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0023 // fingerprint=9be1276dd9547451376d5650c6e20b21d0d2b8d537fc0af91d8656423be54199
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.2.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.input_layernorm.weight')): // bf16<4>[512] name='layer_2_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0024 // fingerprint=13bbefc30cf1c5327f2dd5c4e75420af91c7743506fac1c095bf427ae2fd6b50
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.2.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0025 // fingerprint=28f9af2cc4d43044267db84fcb3432654b3e1c724970c76a8a33cbdd2d53455e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.2.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0026 // fingerprint=b016ef77f48bcd7cfd96078ff22f623efee639247e0302481ed7a5ceb08be748
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.2.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0027 // fingerprint=197c81dcd1c563b3aa6ee1d64d70ebff756d6a0bdbec6fbd88a67090e64fb478
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.2.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_2_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0028 // fingerprint=d7cfe86c58d38affc19a85965ca5b380248fb417219e631bf82dfe0a01badfef
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.2.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_2_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0029 // fingerprint=72f27141e1509270cc284ae2e5468d3e934d2c88e75d75d3dfe875de2ac57231
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.2.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0030 // fingerprint=f45c4763de90c5f2593c72d031d1b9424fa242f1da23e7dd3b870e256be353d4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.2.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_2_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0031 // fingerprint=a28bc5fcad53fd9c421cd0de88b2e40d2157dc06054e85c82361661e84536196
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.2.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.2.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_2_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_2_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0032 // fingerprint=c7a6b4f01d53663390c89344865398fa82fd4b9e8f709552df3c31b31224cc59
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.2.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0033 // fingerprint=ee726f9cfc02bc64a674ae9e93041226fd2c44e67d83ec25912b46b42a1b8a1a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.3.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.input_layernorm.weight')): // bf16<4>[512] name='layer_3_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0034 // fingerprint=6408511fe755b2dfa90903d2cfba5d3af139086d46070e313ab1d75ce8c64ff4
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.3.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0035 // fingerprint=3cfafe203b5e2badba9b73f59b4c2811c8dee3e51faac523f5ac8113be709c31
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.3.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0036 // fingerprint=0a007bbb56bd071d8586d7e7e5b776e14c0ddfdf1f0f03f08f2573f274d28365
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.3.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0037 // fingerprint=4955bb61480ef9e894a2c294cca6429f05d20916c26c9368b31617c640ef3e95
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.3.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_3_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0038 // fingerprint=d255a072d266f87177ea62e37edf357bd8744c86baef0421c8f9085a34971c10
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.3.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_3_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0039 // fingerprint=cf1315cf6cf82588fa0390a6e5299d390081eab50f624c4e6aa2d80353d128b3
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.3.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0040 // fingerprint=29a5f625d750ddb398a4146ec6efc8ea0321e00b146c5959cb44e4f0597e80d0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.3.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_3_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0041 // fingerprint=f48c76d83bb5c316f4207d04c1350da86f78ff95367a96f0f0c11e4fdca0ed36
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.3.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.3.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_3_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_3_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0042 // fingerprint=313070c58375b709ade53b2e298dd8dabd9e7b1532b4c628dfec987f827a4d6f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.3.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0043 // fingerprint=f37cb1f5d569f8559a7f4670935c566a727a1ae2512131fcfedfeae315c4ec41
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.4.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.input_layernorm.weight')): // bf16<4>[512] name='layer_4_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0044 // fingerprint=17df261c826549ed9119f7aaf759b646eeee653f19993bc9f40975c2257cc299
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.4.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0045 // fingerprint=0dc26f8be1436331d0cab4618b1510143c9f6381f53735454627947ca49a2415
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.4.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0046 // fingerprint=4347b71c55f328d30018146f9a0dae32616a9b331cbc09a5a62729297ec64ae2
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.4.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0047 // fingerprint=460722e46646b16a8c5db8778b2a411ce3f48a874d92e70f9aba3fef4c578de2
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.4.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_4_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0048 // fingerprint=d9a0b75d585707cce46dda13947c2317c677262e40482a0b14da239f93a8dabe
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.4.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_4_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0049 // fingerprint=64554e5db3d66cfa5bae2efadd0ec5bdb5ac12c83a0910ea551b01dbf80013c5
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.4.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0050 // fingerprint=b6d772c891be8c74bd19f027f8c51fc014482d6a66460fb6657303280ccad2be
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.4.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_4_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0051 // fingerprint=f70e761099a1721c82c5d10744c86d8a89c430805d57ac37b66a92c2502587da
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.4.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.4.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_4_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_4_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0052 // fingerprint=6662c375fe612dbc0e739d635c06e648fe1ff1dd7d58b3d7741b35ecbb9a9f11
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.4.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0053 // fingerprint=9a68acf27000a27a360131111a0adc9b98cad1b7b70c5446035ab98c26cf4390
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.5.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.input_layernorm.weight')): // bf16<4>[512] name='layer_5_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0054 // fingerprint=a34f55675ec54d4f9f1da9de4a6a7e86cc3910a1c75e25ccf79e4708a772c451
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.5.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0055 // fingerprint=fecfc8aa6ade5e6d9663c02cf49747b6022b18952ad6358e53b7a3d4c617484d
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.5.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0056 // fingerprint=ea507c5094058957e2e40db26097d5b593d632cabaeae8bd625c81891b3d7ebe
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.5.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0057 // fingerprint=b90b568195a38c71be31cfc5a6caea0fceb88e7ce95637f9a1fe30286df387ad
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.5.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_5_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0058 // fingerprint=9d3000114f643236f210cac37304e4d7353e151b49126132dcd27b134190db70
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.5.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_5_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0059 // fingerprint=aab9beb8025d0aaa8e28c991b0f1249433dca2b4c9c6d8338d36ccad26285d93
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.5.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0060 // fingerprint=b9255aff2bebf86f4b53fe66f7b3dba93b5b9e682ce148157744c23068a3bb97
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.5.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_5_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0061 // fingerprint=35a30f2c60e18bf1f59cf7add3a7f60f3219381ed86d153091df04ab56ed140b
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.5.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.5.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_5_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_5_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0062 // fingerprint=6d18c8ed0f0aca032b6643e941aa13b703c3c55a7e6db09f1be27005bf9a5e26
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.5.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0063 // fingerprint=4e8fc19cec440af00d24c4e9581a187030c13bd8c0b5caa8e084e303518222c0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.6.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.input_layernorm.weight')): // bf16<4>[512] name='layer_6_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0064 // fingerprint=e2d7c7f4493f118c31250588efeb7d5baf5c7a3d1e50c9dbf1ed1c3684f4561e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.6.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0065 // fingerprint=044cb0dbae90bb2e25b18f5b2ef6d590540f231eaf9d624aef5dd175c99a41c3
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.6.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0066 // fingerprint=2d6b9b04e93aa8de6c3ac46fcb836726327fb094c7c56c2b26ac37c400c2f8b4
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.6.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0067 // fingerprint=0577db91792abcb27bb51fa24a743b9e0818dfbcb70f5f74dfe0be9a0e9e080b
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.6.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_6_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0068 // fingerprint=4a967107f166e4b76549f61824a61bd9498e610335a89eb72bf7bd111476d51d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.6.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_6_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0069 // fingerprint=78a54038a38e159b3d02d94ca9a8eac7869b19fdabf8451187750cbe6449d2ff
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.6.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0070 // fingerprint=009b644a23deb31b0e0390c45715808252456f84c92d9d760f8b97f10833106f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.6.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_6_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0071 // fingerprint=8047d68dd9f1c9c8d1467224f63d6feb8f0f696ee99ccdb8dc60f032ba5da9f2
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.6.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.6.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_6_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_6_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0072 // fingerprint=65498baa82aacd60d96675ad12fa166c2b2794b54c26b813a614b232058cc573
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.6.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0073 // fingerprint=0fbab2604f45ce20c6c5f0fae0ae3d89b425329341d5e2ac3d6004d4136af233
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.7.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.input_layernorm.weight')): // bf16<4>[512] name='layer_7_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0074 // fingerprint=8079d978d93b3bdca2f69e4829fac912bd77fc86392649c7851ed9ed99994794
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.7.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0075 // fingerprint=3f3bedeb7dfdd0d419a09ce2adf29b33ee3fdcb23cc383a230e48ff47e22a2fc
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.7.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0076 // fingerprint=52c248f150061167c574b722faf59ac2157955a83e886c9c1ee80ef98f1ca565
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.7.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0077 // fingerprint=8c0a8632b11be6789158d244a6b90298b28619a4557c149372fda787e9c6ab0e
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.7.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_7_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0078 // fingerprint=c9d4bd25e3e23bd85d43f36ae21a0d9e012cd2bbdef4ce0252db6e2671d2d3db
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.7.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_7_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0079 // fingerprint=c941ae11fcd6cb910156d3cb4be84b12cb42442ad4b29b689d0f4e49d20320f8
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.7.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0080 // fingerprint=c3fac33adaf5204bc0b359a27aa200c75bb7321ecab5f96ff74f669b05a10b5f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.7.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_7_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0081 // fingerprint=be53754faee8d58569ebdee5821a4c5bf08aa20c68ac14d0e763b8bafc4a7a9d
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.7.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.7.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_7_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_7_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0082 // fingerprint=d57974076ef8e235655b17837e756cf8ab785bb7f95a787348c7fe8c2db90577
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.7.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0083 // fingerprint=e7764b169e98bb5cb4ccd613f5a034796d40377e1caecc46ed78fafd09b937f2
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.8.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.input_layernorm.weight')): // bf16<4>[512] name='layer_8_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0084 // fingerprint=3def634da412294842962ce8efa2afcf79ee3786d08739b0e7e5bf81fc145f62
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.8.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0085 // fingerprint=bcba7082aab55b892cc16d880c86e6866e73c0e68295657a7468af7322cbef67
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.8.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0086 // fingerprint=8de38d12bea418747cb9b2b79cab0984693f978a7ae1d6b7a82137f4f90489ca
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.8.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0087 // fingerprint=fb9f712b861c25f25e09ad1c9d1e93f133f575248de517e32766c83324b96875
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.8.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_8_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0088 // fingerprint=0c4626a386db33681f52ae81d4d016cc47e18dda8486baf4207ae11304cbd84e
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.8.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_8_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0089 // fingerprint=184944f5d6e202b5f38873098c543bb1d4194ad14206f64478ce02ffe78793c6
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.8.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0090 // fingerprint=f8d1078e91ece4501ec3d044da61c8c96a207b6e6cc3ac893be992de7ba888f4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.8.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_8_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0091 // fingerprint=12465258e528a88306439b260fa2df003a1cc69012b1dc41492d5bb266f9697a
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.8.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.8.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_8_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_8_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0092 // fingerprint=903b16f5d8ac791705086b1e53a2003e3ecc41b15aad90a175140096ee419d75
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.8.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0093 // fingerprint=229079af713d5671057215c7330dd20a778d08f874a2dce7bb7bbe00656908a3
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.9.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.input_layernorm.weight')): // bf16<4>[512] name='layer_9_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0094 // fingerprint=73a9985b1deb89645d73b687953d44a77802f2bda47d31bdee8a908d3e1f4e42
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.9.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0095 // fingerprint=d444bd401079c35fb03580ccb35d0d6eb7da7443494cc6eb070f23dce288227d
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.9.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0096 // fingerprint=a32d4357795557bf043024b071390f3047e54cf7f33d7e30c4ccf81c85eb8249
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.9.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0097 // fingerprint=c4f73b41277b71b03287b892646431c6d2436b55b8be29671f475b0f1b27144f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.9.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_9_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0098 // fingerprint=e0075d7f68d04855464be46adc764e3b51281618cdfc0525b90d6b3df597b424
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.9.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_9_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0099 // fingerprint=8b3fb97faa9e21c407a504dae105eeb823b068ef5fef6c41ded18fb7c6407b8e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.9.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0100 // fingerprint=cbc2e099c7cdfb8a650c43dbf75c87ddbd83b89a3a97e4a2fa37fa72b753c6fb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.9.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_9_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0101 // fingerprint=fdfa0b91a534963d41d744edc81d48e972b960ef704f1f0a01620885089a389f
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.9.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.9.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_9_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_9_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0102 // fingerprint=e12f36a0ced28459c705b067d640cab62f036da72735e29addbd7a1ac0a27129
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.9.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0103 // fingerprint=b2b09641050161da11546ecb80fd3fa13c2f873346c60839f512e628420d7644
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.10.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.input_layernorm.weight')): // bf16<4>[512] name='layer_10_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0104 // fingerprint=9d3ff02447c0024eb5caab7e328f621b2b4035b0f34406926aa67f48e0ad2aaa
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.10.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0105 // fingerprint=4b83a343b5119930755b8c6cc82c04b1ebcce20ede3cb5311e1d5f03467b36c9
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.10.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0106 // fingerprint=4db44f88cf2a1d66819bb56693de272e64c700299961b7bbae14c006dbe2de77
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.10.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0107 // fingerprint=5a202ac562df8016e941e3e4b266c963422c05a7174034f55cf8c419388183b6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.10.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_10_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0108 // fingerprint=09346ed2e4e06e887750e91d6a87e44eb80fcb89af1deb5e95106c5d070ae101
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.10.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_10_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0109 // fingerprint=f924a275a607fb2c0f463e630c7429113b9624e48a84b1967f0c59dbc49e81d8
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.10.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0110 // fingerprint=415fa2711d357c9b527f2fa95e8a5694fd8d49adb6bb4980051524ec54c1d534
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.10.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_10_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0111 // fingerprint=d5eed09cde86aeb0bb9158f6cddf7342e2a48f764bae06158626db051aa18e8c
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.10.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.10.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_10_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_10_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0112 // fingerprint=d0435e21661e67399e182edd5b624e31cb1e61bb21406f005b9b3e4e4c1a9c59
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.10.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0113 // fingerprint=37b42f26dea5f1e41531854a3db70369ea3ed591d9f109cb1d6662272841c07b
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.11.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.input_layernorm.weight')): // bf16<4>[512] name='layer_11_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0114 // fingerprint=b65b91a7f2a9a2f6070b03566ecbbb84bcb82785e4ae7dfde942cd987edff622
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.11.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0115 // fingerprint=6148d46843b668bf79cf193a2e64393d65c581e5a4d762cf6af0b0417b4035a6
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.11.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0116 // fingerprint=347cb2a943930f94ebf918939d1b8d39980ceb660ac624ab88b155dd92a51369
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.11.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0117 // fingerprint=129dedf6e54b21d30f2693f914e4f4ce5d0c9cdd700e16840749015d3254cf5d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.11.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_11_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0118 // fingerprint=4bf78a1563a8d71359d4df897ad43eb48d8971526f08214e34998ce3ec27bd4c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.11.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_11_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0119 // fingerprint=c82749a60a6b3f143e29765118c144614f87b01bf95c9a27bf13349c45e1b431
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.11.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0120 // fingerprint=ba116a5948b347944979500dcaf6e6ca5f78d883206494f16ec5d9cbbd5d1a53
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.11.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_11_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0121 // fingerprint=cfa51f080a1d9c0b872ce913478c823e6869c1c18b8faedf6c47a9abf86e56cd
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.11.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.11.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_11_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_11_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0122 // fingerprint=ddb9cf219f4aac7afb12e4ea157c0d4538a8cfc16802cb4be30ca200c15e5ca7
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.11.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0123 // fingerprint=35e46f54cc63c86dadc83df5fbb9e3753255517dc15a3f2a0fc8cfeb95cc71ba
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.12.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.input_layernorm.weight')): // bf16<4>[512] name='layer_12_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0124 // fingerprint=ec13af8febb238415f0a5f5b58919f56c4ddfb1e1d026908ef2145ebce669588
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.12.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0125 // fingerprint=cc23c0c4255f4c3fc37b3fefde133b5f7bea9bc55772f62898c3f3082ea82e9d
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.12.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0126 // fingerprint=de4a92b7c13825e97f0426616c880263aa69bfdc96a5f300b1aae8c8ca5547f7
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.12.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0127 // fingerprint=1f7dad6ec18545b29cc44c7b22b7d8f949f1cd8f25e9be1bcec67b4d795e18cc
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.12.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_12_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0128 // fingerprint=6eff981a16eeadab9bcd3f0118fe5ece1039bffc61d92dd70f08c4fa5adc2f27
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.12.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_12_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0129 // fingerprint=b5f7361e4747304af4a19d73f64872aa218ad0908588020a92f7efd7128fd780
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.12.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0130 // fingerprint=de9557f2cc522287adb97eb53b965929a304e99cf36e3f2028995aac09efca3d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.12.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_12_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0131 // fingerprint=fe749b12d6be3aca78fef23fa4a210af8d6689ccb954df1b8ab17c7ceb696fe4
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.12.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.12.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_12_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_12_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0132 // fingerprint=f45aef2492d450ac4e757ae1038968c07bbd6d1b9c501f65162e876b62dd5167
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.12.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0133 // fingerprint=7c2bcbeab1f8bbc8ce28d3b7f34b64129471f2690e7b6c72764f53414c776f43
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.13.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.input_layernorm.weight')): // bf16<4>[512] name='layer_13_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0134 // fingerprint=4c1e32f376053e15e8a63a7a7d3e063e813d0f75a43d82e8e6d70302838b2537
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.13.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0135 // fingerprint=7c744cc1e2e3f45c4957b2012d6996fee900b158c75832af3e69c66db9b0e985
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.13.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0136 // fingerprint=6ff0a58c804c0a6045dd74642461662b8a0d79e58f9ab4fb443294eb54ca531a
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.13.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0137 // fingerprint=06b4cd66e62147d4cbb6558e66a5ff81a584bff04ed167950b062054227d28cc
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.13.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_13_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0138 // fingerprint=e3a66615e4179308c5c4c66840442b9dc8cba943630b3b614c2a9189af490097
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.13.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_13_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0139 // fingerprint=0921c2a1cb61686cd99c0a07ffdb0d7683ba3f12aa76cd39c1fbe571174ad091
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.13.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0140 // fingerprint=606b3439e108938cc9cde4c9183f49c2c993815e47a06f3e52ff7c6e898b024f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.13.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_13_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0141 // fingerprint=a39adcddb5576663b3f9117f42a848d18596935ba110b03deb72bbeb6a83bf6c
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.13.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.13.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_13_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_13_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0142 // fingerprint=8e2b61e289d387a6f6ad713641f0bd45ef4041b88b1ae1a05dab3e9950ab1012
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.13.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0143 // fingerprint=37140d56722be2a6f9e0a1d838d35bcf1c06f867b87fccfeb0263a613e9cb770
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.14.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.input_layernorm.weight')): // bf16<4>[512] name='layer_14_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0144 // fingerprint=83a37291fbf399a81b1258fd41bb104f37fd32f7417bcf96a9f74a5942ce96e3
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.14.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0145 // fingerprint=c17cb335296e180868b189c9c7c4b51612d97a03ed9fc39f8a599fc5bcb8fdb0
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.14.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0146 // fingerprint=bdb5de328fe06b075daa351b9aee13a7fe5638db2f238de45ee7e246d0c254bb
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.14.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0147 // fingerprint=12c3d7a4e62ec0db23dfa9ea2ee7636677e3fac4d0ff0c783344f9eb1cf6b071
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.14.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_14_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0148 // fingerprint=11a99955d742817f1eb948150372479a0917b76709870111ffab2fd8d53b21f4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.14.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_14_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0149 // fingerprint=30e35f386ccdcef807b08ed491ee1bf30ad5c40000a103db1f169c4bd92f4805
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.14.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0150 // fingerprint=a465dcb3ffddf62d948053057c9a9bc7165148d83d323e1d5d3464006f007136
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.14.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_14_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0151 // fingerprint=5040ecb890cbc01397871d10b11fec3e50a155b315a63f3a1b8e38c7ab43109b
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.14.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.14.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_14_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_14_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0152 // fingerprint=b4a7a3ac041fa46179c5fa5653e0385567c6b80153ef1faf8fba8b05a19ae949
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.14.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0153 // fingerprint=54b8b830eb9df57a97eee0e89135e26c6e64de125d5047c1ed8dab2cbe933e6b
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.15.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.input_layernorm.weight')): // bf16<4>[512] name='layer_15_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0154 // fingerprint=b78a056a901348c92c230d088dac7c6e78bfa014479e597e4b4c8d28b1631410
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.15.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0155 // fingerprint=0caee1b55fe332f2eba3d4344b1c198faccdbbb07f6df2a5e1880ec443ba5456
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.15.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0156 // fingerprint=8db481b478be216c1b195e518179a439b0428e8cf58ee95dea8bcd7015cd0e30
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.15.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0157 // fingerprint=c2181143ccd5d066a27739f4be607247bba68e09b6e896b2721b2e370e6ff29a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.15.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_15_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0158 // fingerprint=181d6c7739fc05be819977fd646e69c12ff2a02d37253cab93208dd9a998b495
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.15.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_15_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0159 // fingerprint=40ef65e48a5d78ff556bb98f7a639ea3498103d67ec8a5114c0bbb9898a51062
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.15.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0160 // fingerprint=3e104cd1d8594a6f3a22e31bb832d51d2453323dc27ee6c62f00ca9f370a8aab
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.15.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_15_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0161 // fingerprint=668ec03d249698222a252577c18eaca9069442a53279ff5f32a042e0c02d0ee6
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.15.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.15.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_15_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_15_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0162 // fingerprint=3e141b3acb4da8f6c605e569088dd37795ab3067b51e0c1f6e090052d2672716
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.15.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0163 // fingerprint=89ccc442fb8d7bc8f16f3e89899a50f10c3712cb8fdf383a09a084bd4f0b040b
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.16.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.input_layernorm.weight')): // bf16<4>[512] name='layer_16_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0164 // fingerprint=a47090811d106c6e563da8172bae366462589af9f1bb0b301ad3b7ca2e43bd6e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.16.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0165 // fingerprint=63f9866fba202ff8fcb577b1ffe452a96ca2eea9cf9e12908d65c48de4cd4c48
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.16.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0166 // fingerprint=c9f9360061ac0e3a9bd0cceb16aa1ec280e9927b7034bcd9241fa02036f4c9e2
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.16.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0167 // fingerprint=e0925a2196c2450c5431243419bd42f70d3101ce62154eeff1ebd985ba3a8029
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.16.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_16_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0168 // fingerprint=cf17140fefe32530c4a9b64cccf8b5365b70248ef6769b1fa7c346fc997f726a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.16.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_16_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0169 // fingerprint=7a08e1f049d85b7a83b916f00bee622dcf677fcbfd10fdde091f3d8b721b9013
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.16.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0170 // fingerprint=484b4bf39276a4870994ec299b6672a497e94f8c02a9f54ebc3290f34e7683cf
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.16.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_16_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0171 // fingerprint=0bd90c04905ba8b8be3ce6b9ca6ccc49886317ad273e25db8d5e4608f7ea6f42
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.16.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.16.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_16_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_16_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0172 // fingerprint=d7511a1c989459433be7da92b055133f0aaeffc5b716e23704d934f1951a3460
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.16.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0173 // fingerprint=bb33d930cadb51249b71a189350c878dcfaec127d86cde402bb1b08fa3d923ed
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.17.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.input_layernorm.weight')): // bf16<4>[512] name='layer_17_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0174 // fingerprint=f2a34a8994106a9da381539190a195e8866ad94a73b41e5c5a917a7ec7bcf34d
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.17.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0175 // fingerprint=c9ae5a793e0b92245ddc0e5c3693ef7ff205668debc16c43fb9c5d91653b0e62
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.17.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0176 // fingerprint=5c2f593ff4d55054e4d44f1ddd0986d54969198cc56afdb73c1b6349f9f599c6
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.17.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0177 // fingerprint=50b572683e8efadb11984390778bc34d1d6955112d71083611e2f9943b7f4ab6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.17.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_17_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0178 // fingerprint=6e65f10deb147c23d0e66c5b17e13d51dfed0c0f1ee619e99e3099cecb5340c7
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.17.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_17_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0179 // fingerprint=e711c5f48ba0129d055014fcdb46cd77381a03fcf0202ec6390b682c9f78f5a2
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.17.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0180 // fingerprint=fe11201ea56931d200c6339aaf3d3a4df46c44db44a96356563b188f109f9f0f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.17.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_17_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0181 // fingerprint=8f24dcf9a1ddef7fcb6eba7694e2fda7b4f80d5222fe9072f8456dfd44d092af
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.17.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.17.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_17_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_17_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0182 // fingerprint=7ebb5680261956319a4927b742095936a6490e143bd0a5c4b9f9782c495f92b4
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.17.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0183 // fingerprint=134bc87c450d62a7da1f6e97b3489086317809daab977d87b8092c575318e33f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.18.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.input_layernorm.weight')): // bf16<4>[512] name='layer_18_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0184 // fingerprint=c08a406d78828fd6e8d2f4bdb9a853d76a91748f97811a6349d55d700aa8f02b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.18.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0185 // fingerprint=52201c9395ec76c641343d23de8a26eee0abe2fabb6b188686957c35fadf9a90
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.18.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0186 // fingerprint=ed12aa5c2941aa8bbb008811648c1a5328f8bfcac3a581b7fd7c9ebde1e35dad
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.18.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0187 // fingerprint=cd671a8e660612fa8f7c31d74f610aec139719d23b81cbfb78c80ca8126b4958
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.18.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_18_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0188 // fingerprint=bbcaf548ef156dd181befef1a2f908293392c896f8caa831523fdc85ab60f7a4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.18.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_18_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0189 // fingerprint=337d93a393ea0e3e4f495416c1fb817cc0b1e905f2f2763474b7730372289ee2
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.18.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0190 // fingerprint=5ca5ee41ece7a974f5f6182150a53e53721cf97f9a6d6a8a2f3a7557982b7e59
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.18.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_18_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0191 // fingerprint=f2ca915392c10b9438dcfa4b7bc0882c39f08d1a5aef6329e9c74e56121bbc6b
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.18.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.18.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_18_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_18_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0192 // fingerprint=5f9ee2b7302884da12320055c8e959ed20d514c4497d8924df3ff87e3157c082
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.18.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0193 // fingerprint=0fffea7c8d90f54ff83402002a76cb0d278a77f4d22cb823275d808482f2c3e8
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.19.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.input_layernorm.weight')): // bf16<4>[512] name='layer_19_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0194 // fingerprint=5f5cce1555d0cffe6a77c94c169b0567505e6c2699f320b2ef8b4aea8ae3e5a0
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.19.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0195 // fingerprint=e9b4a3d047f0fcedc768a7b1ff23acf090f907de05979e1f8ff33a7cd7f6c3e2
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.19.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0196 // fingerprint=0f082c82cb2fd24fdd7a5eda60e80c2803e14a52162990b1b2bdacbad1bbf238
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.19.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0197 // fingerprint=e0b5d9acd79fbad6cadc19f458ebfce2d09f03c5c973f223cf90361470cdee3f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.19.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_19_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0198 // fingerprint=8ae05bf8f401b8684c154739e8ce7338523ebf71012455cd9ae2cd47dee4ba9e
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.19.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_19_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0199 // fingerprint=dbd8626e77e75a65ff339188723c805e3569c9ae1ea68a8baaa941f362a44497
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.19.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0200 // fingerprint=fc05dc4e9d1749b1dbf13872bbf7b932c67c1801c855e45f4cad7f3fffc1d41d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.19.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_19_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0201 // fingerprint=cfec13f8e48e73e04e9a7d38685cd4b12d8c8dcffe28ca359e3da5adf9410804
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.19.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.19.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_19_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_19_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0202 // fingerprint=352c59e7203d57575710fa9ac8d4325bfbe2725e2ffd02de20aae833fcfcb646
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.19.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0203 // fingerprint=9423e6f202e3bcbb9b010e9d333c2a85d33939ab3c4bea3778020cae9808a9de
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.20.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.input_layernorm.weight')): // bf16<4>[512] name='layer_20_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0204 // fingerprint=d4cf0e34ef8193215acf8e597e6a7339658725569788fd062ab861b9000bc0dd
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.20.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0205 // fingerprint=9cd3579163fb2821451e05c2d17baf08cd6b5769d2c47e894220cdff80a74d9f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.20.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0206 // fingerprint=1aa9c225c7ce145f475db454ab5316f3e84ca18341a4a643cd64c0e5610d2aea
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.20.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0207 // fingerprint=8b9e190beadec7c81b7a108b0519ec1211f60e0d248ecfd3e95d6e74439dd4ea
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.20.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_20_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0208 // fingerprint=90280b76ce0afb75bf20c6d7262b6bd608be43775b5f46a0bc0bdcc668678aa1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.20.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_20_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0209 // fingerprint=1ad08ca2dde7faf7ab13507bca59f527c8664a645a57022e3ded778fc2d5314c
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.20.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0210 // fingerprint=27fac0a796d3a767f22a1ce131796d63252873893b5ddd583b1ff7ad1578673f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.20.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_20_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0211 // fingerprint=bba650bc6998b7da036efc32b3eb54653cb900c588fd2e9ac163627749e1296a
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.20.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.20.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_20_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_20_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0212 // fingerprint=724262a734238a3252ad99419de64c5e60e1491fcdc1a920339fb82dad07c5f0
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.20.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0213 // fingerprint=b5e32db7d45ad15d3501b2a3af8e1642d447d77e0dbf0428f9362a90f5139516
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.21.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.input_layernorm.weight')): // bf16<4>[512] name='layer_21_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0214 // fingerprint=cca92e2d2f03443912ad7471fb3c37a16dc870aadfc22966682bcfe9cd18b163
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.21.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0215 // fingerprint=587644b216ea80a80309abff5f6b12f61ae4b4c58a49abf68671b94d73e7577e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.21.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0216 // fingerprint=d7f83652972b008fc02903a5cff48213f8cc0185faa078d814684e18c579fa96
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.21.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0217 // fingerprint=b1250e7538bcdc3dca5a8eb698492a36e5cbf00e4a26fd5ebb7403a248c350e8
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.21.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_21_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0218 // fingerprint=d6857d8b31432860740ae7fdf8752559b1a08d64003108f6ba76b6eeb0eb10df
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.21.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_21_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0219 // fingerprint=28c25b14d74f0f5e20e5f905fad7d7550f44fe816bd9457e8d84c99bb176f242
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.21.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0220 // fingerprint=abfb9c2b35882ea71186fe4efe5744c09c35c9226c667aef2f0135f26a43334a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.21.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_21_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0221 // fingerprint=1fc96f568341cda222d683aed3fdee0ee2fee44245b8f83768a3b9cef192dc9b
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.21.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.21.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_21_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_21_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0222 // fingerprint=febf6a6919e35bc726e0a094d58a56b9a37efb91481d556e9f09612e91489441
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.21.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0223 // fingerprint=ff443fac360e1c31f8eb2e220ae562d713848381245a8b3eaf13156e497d874f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.22.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.input_layernorm.weight')): // bf16<4>[512] name='layer_22_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0224 // fingerprint=ab8c89b60d1d5bd7cdd2de31489f2a316e28c3ba9ac4e150b25a43607bb10d64
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.22.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0225 // fingerprint=47a76c35755d87dd95cfb12e0f9589403c31e9e5cdea8a5103f722523948f07c
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.22.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0226 // fingerprint=ca13e686d0b490af7b6c6a2852bb2854f1b662b409e3991d8493ef1c76277b5a
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.22.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0227 // fingerprint=db3c4a516dd63688fd9e8e5bc545387347cad6e3b2484d03c061922c996e7e27
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.22.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_22_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0228 // fingerprint=ea088d95584946e1251a0039d7da5ad6001b0baa0ca54af62f5088235fc28bbb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.22.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_22_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0229 // fingerprint=ebe71945fad90d75ad7d6953e4ae1b87e0c3069be40f148fb1789183014366d4
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.22.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0230 // fingerprint=18f906752846de1343d0e45ad2e6790f924f37b590cb7f150babb47d24ac53fe
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.22.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_22_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0231 // fingerprint=48aef8af2f0e0eea2c194b22a45ff246ca4c7d2a218dd89247eb280a5f27e1ac
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.22.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.22.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_22_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_22_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0232 // fingerprint=02c2eb0ff41be50cf7405941769834eb24230208c0ec08df663b99cd5c1cb7e7
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.22.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0233 // fingerprint=484ecb72a36404f4ab0a4697f3ac548ff880ea6f97dd3278a0f24738333292de
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.23.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.input_layernorm.weight')): // bf16<4>[512] name='layer_23_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0234 // fingerprint=25efe231e26d88a714e621816576bcef82bfda420f0ead61b9e5a6c1d5ae863b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.23.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0235 // fingerprint=2f73a4c4425c8e2b8dd12d973cb33ce45c44913581a1fb7723c69cad315b5f2b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.23.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0236 // fingerprint=e5100429d76f67095e689d1833e991b6fd71eb6670999eea2fe563a93f658fc7
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.23.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0237 // fingerprint=2bce1eb661375df5c3bfb2e406ed12c9311d4bfce13ea522035a7a8665288b17
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.23.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_23_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0238 // fingerprint=9237aee36a7a4635446584cf0f50da07ea1bc396dddfa5e1ae29abb9d3796b85
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.23.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_23_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0239 // fingerprint=dc48e4348a0b5aab4643ff83b030e462701df7357dcca0d5af259570cc5b7526
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.23.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0240 // fingerprint=dea4fc2931a6a532111d87ac1441c8786b3725c0b8c10131a7a13482c855cd84
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.23.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_23_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0241 // fingerprint=8ca99e4bd48e1bfcae963e91d442f6324882d9267b9d9479615f11d1cee3bce8
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.23.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.23.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_23_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_23_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0242 // fingerprint=3a359b6b6bb6bb4f12dcef95a20d0066cb2ec59ab2b4c08c9083294ad6c805c6
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.23.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0243 // fingerprint=203159aca5d96919da7f0908a9283eab30b11ef09676f529f3ec185c9d7d94f7
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.24.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.input_layernorm.weight')): // bf16<4>[512] name='layer_24_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0244 // fingerprint=3543cb28e278fc37a31c184b0c43e538ed077b36c28ea32184bdf1c2394be6f1
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.24.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0245 // fingerprint=705f05756fca58bd5f10cad72d72f7c7fc2fdc35b95eed36a7a29490b408e480
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.24.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0246 // fingerprint=81b978df052220c0bdb3951d2358e4f37da97c1a1e71048db3f0265eb98dca6e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.24.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0247 // fingerprint=9b72f95b15869753d9fe7ec9bb82008d04f437c16817f6647e9a4a05d2b550db
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.24.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_24_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0248 // fingerprint=7fcdef3c228a247ac38ce5942325637536f372318860ba69f0507becfbbf1a67
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.24.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_24_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0249 // fingerprint=98e5175f7a832540da87e6e4f85e2c21f4e31ead5f471f896fb16230f3272969
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.24.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0250 // fingerprint=95b357ab5c56921a1bc6698574ebab66bb8fca3a32124bd0fa2522581c169f22
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.24.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_24_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0251 // fingerprint=6c3cec95f1a6026df1806c450c00b94a4b2493510c87a83b33a060906f5b59eb
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.24.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.24.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_24_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_24_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0252 // fingerprint=69f9af4c1bd7d48d5ce0ff5f52361394665c298519d4ec59671b5b6e07fba68f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.24.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0253 // fingerprint=930a0be82dd26fb3eb0d9a288680b74f71ed0fa1c6e5d2f39220b9d059330a6b
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.25.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.input_layernorm.weight')): // bf16<4>[512] name='layer_25_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0254 // fingerprint=a06516b88845156d116f1184206ced9967ba655d08365c94497cb3d220e3abe6
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.25.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0255 // fingerprint=599524b753607c8206387b4dfb9ab759b86c17f422163279cfd81d980385ce0a
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.25.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0256 // fingerprint=f2dd01d16c4b443deb95067d3a18fb72126f7167750344e4e20c20b6b8353b90
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.25.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0257 // fingerprint=b4cb5d50e7d1a87e2dddd505afe423211f21113b19d4d5b6b1bff7143273a665
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.25.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_25_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0258 // fingerprint=ca836feba75ded27a1f26d0ca1d667b9c4e6ba7e7607eebdedc03027ae85d436
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.25.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_25_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0259 // fingerprint=bf1d72f394bfadb70bb58b07cec62fe9ecf1712ca9f62a45e4318b5074d2f270
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.25.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0260 // fingerprint=508653f70f7362cfbad13cf7a2f5a3c24869a2fb4a74e9cc4f8bc9b4cc3ded93
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.25.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_25_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0261 // fingerprint=c7dd44ccc7bd53797dfe2cb3f239cacfd4bfefbfd457e22e8f79560a4a61e763
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.25.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.25.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_25_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_25_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0262 // fingerprint=7e7b1a9f57826c7eb4f4e109086342a892666099387897321c9f4d2c5681e1db
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.25.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0263 // fingerprint=23c4e1162c93ece82ec78f80038c2d9ab2d1b18a0a545889efcaa95e212301ae
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.26.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.input_layernorm.weight')): // bf16<4>[512] name='layer_26_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0264 // fingerprint=bfd95efb9f036c3a8198ced29c9945181c683d03094f4d2d8f5086d4b1af5ef0
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.26.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0265 // fingerprint=3b219cfbf14e2fd8b26d389428601ba333c0cbc2d5b2806b722ee0dd0f4eb618
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.26.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0266 // fingerprint=495b98e99a39e1116c8221cf488372f23fd08f6980c9885ca6312b623e0e98dc
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.26.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0267 // fingerprint=45961c02d352615f7eedf501e471bd85cd541d4b599bbf007333f3bfe873b80d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.26.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_26_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0268 // fingerprint=c78239d7b6f8b20f2d555288edf2296aec7f9183da660ecb7787e3acac338623
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.26.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_26_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0269 // fingerprint=fd22b18ffca512395bbd352069b3b15b0878cc4f636f5b4ef4b2642df9f8fd58
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.26.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0270 // fingerprint=b6470b64b9433b8c5ed20ddd87766acb1906d8f5b65927c52e30a46d2a95cc55
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.26.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_26_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0271 // fingerprint=2ec7f3e0ab46ad4fde75d08c5d299a2e276feb3b3cc6bc7f532e7716a0805fc4
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.26.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.26.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_26_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_26_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0272 // fingerprint=cc59516b241f2333c70de6943630efb2418532624c36e41db0bca014adda9119
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.26.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0273 // fingerprint=8c3f527b73b731a2d1c7a5752df7cf905118bcccb2ada4793aff025b2bbe1056
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.27.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.input_layernorm.weight')): // bf16<4>[512] name='layer_27_decode_layer_call.arg2.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0274 // fingerprint=df91aab312cbaf551f11b7dcc8ad4a926b6b05a30e9590ce866493ebbaac6a2d
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.27.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0275 // fingerprint=ad313d8a9d2934ecc35b528ecd8a1a156df3604ac9b785f808b52f562392d782
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.27.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0276 // fingerprint=eae72f0b128b8b66a7917e6e5d38e32868f1b4b75d7726d2a309bf6b4a95c429
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.27.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      yield (%5)
    }
    constant_recipe constant_recipe_0277 // fingerprint=893e0464e67ba1001b88adba2fc42cb080a7174dfe4080145016b75f890b5572
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.27.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_27_decode_layer_call.arg6.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0278 // fingerprint=bd38bb8bf460db2f1854e4d883ab87759a17e530578aeda74798f7f3a083fd55
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.27.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_27_decode_layer_call.arg7.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0279 // fingerprint=83b0f9f50dc7232ad9191f89baf85ef61b65a72686e0fb0bf7c09dd44052eb6f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.27.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0280 // fingerprint=04e61a86e748604f825b32671689e83fda489e90d0a030acd0d8ad74b3763bff
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.27.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_27_decode_layer_call.arg9.boundary_pack'
      yield (%0)
    }
    constant_recipe constant_recipe_0281 // fingerprint=83b25584cc1d0525e986a79a8306207bf96a7cfdf5e727d1865d9a8ad372bd17
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.27.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.27.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_27_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_27_decode_layer_call.gu_weight_rowmajor'
      yield (%1)
    }
    constant_recipe constant_recipe_0282 // fingerprint=731966290b4d1caf960f087adc41b253906be79a900d6c2cd14a650d2c4f81d4
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.27.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      yield (%3)
    }
    constant_recipe constant_recipe_0283 // fingerprint=ea7af6540fce2bcbf86d0a59228e184aa53df283ca8e092da1fe17bb458bfdd7
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.norm.weight')): // bf16<4>[512] name='final_norm.vectorized.pack2'
      yield (%0)
    }
    %w0 = None(): // none name='decode_layer_qkv_none'
  }
  // compute
  %0   = NN.Embedding(PaddingIdx: None, PostOps: [fusion(value: bf16<4>[1,512]) => NTT.VectorizedCast(value, NewType: {elem_type: 'float32', kind: 'vector', lanes: [4]}, RoundingContract: 'preserve', VectorizeAxes: [1])], %input_ids, ConstAssetRef(bf16<4>[151936,512], Recipe: 'constant_recipe_0000', Output: 'token_embedding.packed_weight')): // f32<4>[1,512] name='token_embedding.packed'
  %1   = NN.RotaryEmbedding(AttentionScaling: 1.0, HeadDim: 128, OutputDtype: 'bfloat16', OutputLanes: [2, 8], Theta: 1000000.0, %0, %paged_attention_kv_cache): // (bf16<2,8>[1,1,8], bf16<2,8>[1,1,8]) !read<paged_attention_kv_cache> name='main_rotary_embedding.packed'
  %2   = Tensors.Bitcast(Dtype: {elem_type: 'float32', kind: 'vector', lanes: [4]}, %0): // f32<4>[1,512] name='layer_0_decode_layer_call.arg0.boundary_pack'
  %3   = GetItem(%1, 0): // bf16<2,8>[1,1,8] name='layer_0_decode_layer_call.arg15.boundary_pack'
  %4   = GetItem(%1, 1): // bf16<2,8>[1,1,8] name='layer_0_decode_layer_call.arg16.boundary_pack'
  %5   = NN.NormStats(Axis: 1, UseMean: False, %2): // f32[1,1,1] name='layer_0_decode_layer_call.norm_stats'
  %6   = Call(Callee: 'decode_layer', %2, %paged_attention_kv_cache, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0003', Output: 'layer_0_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0004', Output: 'layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0005', Output: 'layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0006', Output: 'layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0007', Output: 'layer_0_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0008', Output: 'layer_0_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0009', Output: 'layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0010', Output: 'layer_0_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0011', Output: 'layer_0_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0012', Output: 'layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 0), const(bool[] : False), %3, %4, %5): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_0_decode_layer_call'
  %7   = GetItem(%6, 0): // f32<4>[1,512] name='layer_0_decoder_output.boundary_packed'
  %8   = GetItem(%6, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_0_updated_state'
  %9   = GetItem(%6, 2): // f32[1,1,1] name='layer_1_decode_layer_call.norm_stats'
  %10  = Call(Callee: 'decode_layer', %7, %8, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0013', Output: 'layer_1_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0014', Output: 'layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0015', Output: 'layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0016', Output: 'layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0017', Output: 'layer_1_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0018', Output: 'layer_1_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0019', Output: 'layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0020', Output: 'layer_1_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0021', Output: 'layer_1_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0022', Output: 'layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 1), const(bool[] : False), %3, %4, %9): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_1_decode_layer_call'
  %11  = GetItem(%10, 0): // f32<4>[1,512] name='layer_1_decoder_output.boundary_packed'
  %12  = GetItem(%10, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_1_updated_state'
  %13  = GetItem(%10, 2): // f32[1,1,1] name='layer_2_decode_layer_call.norm_stats'
  %14  = Call(Callee: 'decode_layer', %11, %12, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0023', Output: 'layer_2_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0024', Output: 'layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0025', Output: 'layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0026', Output: 'layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0027', Output: 'layer_2_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0028', Output: 'layer_2_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0029', Output: 'layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0030', Output: 'layer_2_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0031', Output: 'layer_2_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0032', Output: 'layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 2), const(bool[] : False), %3, %4, %13): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_2_decode_layer_call'
  %15  = GetItem(%14, 0): // f32<4>[1,512] name='layer_2_decoder_output.boundary_packed'
  %16  = GetItem(%14, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_2_updated_state'
  %17  = GetItem(%14, 2): // f32[1,1,1] name='layer_3_decode_layer_call.norm_stats'
  %18  = Call(Callee: 'decode_layer', %15, %16, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0033', Output: 'layer_3_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0034', Output: 'layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0035', Output: 'layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0036', Output: 'layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0037', Output: 'layer_3_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0038', Output: 'layer_3_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0039', Output: 'layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0040', Output: 'layer_3_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0041', Output: 'layer_3_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0042', Output: 'layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 3), const(bool[] : False), %3, %4, %17): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_3_decode_layer_call'
  %19  = GetItem(%18, 0): // f32<4>[1,512] name='layer_3_decoder_output.boundary_packed'
  %20  = GetItem(%18, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_3_updated_state'
  %21  = GetItem(%18, 2): // f32[1,1,1] name='layer_4_decode_layer_call.norm_stats'
  %22  = Call(Callee: 'decode_layer', %19, %20, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0043', Output: 'layer_4_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0044', Output: 'layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0045', Output: 'layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0046', Output: 'layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0047', Output: 'layer_4_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0048', Output: 'layer_4_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0049', Output: 'layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0050', Output: 'layer_4_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0051', Output: 'layer_4_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0052', Output: 'layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 4), const(bool[] : False), %3, %4, %21): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_4_decode_layer_call'
  %23  = GetItem(%22, 0): // f32<4>[1,512] name='layer_4_decoder_output.boundary_packed'
  %24  = GetItem(%22, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_4_updated_state'
  %25  = GetItem(%22, 2): // f32[1,1,1] name='layer_5_decode_layer_call.norm_stats'
  %26  = Call(Callee: 'decode_layer', %23, %24, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0053', Output: 'layer_5_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0054', Output: 'layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0055', Output: 'layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0056', Output: 'layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0057', Output: 'layer_5_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0058', Output: 'layer_5_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0059', Output: 'layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0060', Output: 'layer_5_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0061', Output: 'layer_5_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0062', Output: 'layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 5), const(bool[] : False), %3, %4, %25): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_5_decode_layer_call'
  %27  = GetItem(%26, 0): // f32<4>[1,512] name='layer_5_decoder_output.boundary_packed'
  %28  = GetItem(%26, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_5_updated_state'
  %29  = GetItem(%26, 2): // f32[1,1,1] name='layer_6_decode_layer_call.norm_stats'
  %30  = Call(Callee: 'decode_layer', %27, %28, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0063', Output: 'layer_6_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0064', Output: 'layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0065', Output: 'layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0066', Output: 'layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0067', Output: 'layer_6_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0068', Output: 'layer_6_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0069', Output: 'layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0070', Output: 'layer_6_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0071', Output: 'layer_6_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0072', Output: 'layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 6), const(bool[] : False), %3, %4, %29): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_6_decode_layer_call'
  %31  = GetItem(%30, 0): // f32<4>[1,512] name='layer_6_decoder_output.boundary_packed'
  %32  = GetItem(%30, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_6_updated_state'
  %33  = GetItem(%30, 2): // f32[1,1,1] name='layer_7_decode_layer_call.norm_stats'
  %34  = Call(Callee: 'decode_layer', %31, %32, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0073', Output: 'layer_7_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0074', Output: 'layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0075', Output: 'layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0076', Output: 'layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0077', Output: 'layer_7_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0078', Output: 'layer_7_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0079', Output: 'layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0080', Output: 'layer_7_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0081', Output: 'layer_7_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0082', Output: 'layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 7), const(bool[] : False), %3, %4, %33): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_7_decode_layer_call'
  %35  = GetItem(%34, 0): // f32<4>[1,512] name='layer_7_decoder_output.boundary_packed'
  %36  = GetItem(%34, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_7_updated_state'
  %37  = GetItem(%34, 2): // f32[1,1,1] name='layer_8_decode_layer_call.norm_stats'
  %38  = Call(Callee: 'decode_layer', %35, %36, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0083', Output: 'layer_8_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0084', Output: 'layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0085', Output: 'layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0086', Output: 'layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0087', Output: 'layer_8_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0088', Output: 'layer_8_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0089', Output: 'layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0090', Output: 'layer_8_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0091', Output: 'layer_8_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0092', Output: 'layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 8), const(bool[] : False), %3, %4, %37): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_8_decode_layer_call'
  %39  = GetItem(%38, 0): // f32<4>[1,512] name='layer_8_decoder_output.boundary_packed'
  %40  = GetItem(%38, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_8_updated_state'
  %41  = GetItem(%38, 2): // f32[1,1,1] name='layer_9_decode_layer_call.norm_stats'
  %42  = Call(Callee: 'decode_layer', %39, %40, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0093', Output: 'layer_9_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0094', Output: 'layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0095', Output: 'layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0096', Output: 'layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0097', Output: 'layer_9_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0098', Output: 'layer_9_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0099', Output: 'layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0100', Output: 'layer_9_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0101', Output: 'layer_9_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0102', Output: 'layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 9), const(bool[] : False), %3, %4, %41): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_9_decode_layer_call'
  %43  = GetItem(%42, 0): // f32<4>[1,512] name='layer_9_decoder_output.boundary_packed'
  %44  = GetItem(%42, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_9_updated_state'
  %45  = GetItem(%42, 2): // f32[1,1,1] name='layer_10_decode_layer_call.norm_stats'
  %46  = Call(Callee: 'decode_layer', %43, %44, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0103', Output: 'layer_10_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0104', Output: 'layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0105', Output: 'layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0106', Output: 'layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0107', Output: 'layer_10_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0108', Output: 'layer_10_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0109', Output: 'layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0110', Output: 'layer_10_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0111', Output: 'layer_10_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0112', Output: 'layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 10), const(bool[] : False), %3, %4, %45): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_10_decode_layer_call'
  %47  = GetItem(%46, 0): // f32<4>[1,512] name='layer_10_decoder_output.boundary_packed'
  %48  = GetItem(%46, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_10_updated_state'
  %49  = GetItem(%46, 2): // f32[1,1,1] name='layer_11_decode_layer_call.norm_stats'
  %50  = Call(Callee: 'decode_layer', %47, %48, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0113', Output: 'layer_11_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0114', Output: 'layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0115', Output: 'layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0116', Output: 'layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0117', Output: 'layer_11_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0118', Output: 'layer_11_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0119', Output: 'layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0120', Output: 'layer_11_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0121', Output: 'layer_11_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0122', Output: 'layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 11), const(bool[] : False), %3, %4, %49): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_11_decode_layer_call'
  %51  = GetItem(%50, 0): // f32<4>[1,512] name='layer_11_decoder_output.boundary_packed'
  %52  = GetItem(%50, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_11_updated_state'
  %53  = GetItem(%50, 2): // f32[1,1,1] name='layer_12_decode_layer_call.norm_stats'
  %54  = Call(Callee: 'decode_layer', %51, %52, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0123', Output: 'layer_12_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0124', Output: 'layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0125', Output: 'layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0126', Output: 'layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0127', Output: 'layer_12_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0128', Output: 'layer_12_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0129', Output: 'layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0130', Output: 'layer_12_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0131', Output: 'layer_12_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0132', Output: 'layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 12), const(bool[] : False), %3, %4, %53): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_12_decode_layer_call'
  %55  = GetItem(%54, 0): // f32<4>[1,512] name='layer_12_decoder_output.boundary_packed'
  %56  = GetItem(%54, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_12_updated_state'
  %57  = GetItem(%54, 2): // f32[1,1,1] name='layer_13_decode_layer_call.norm_stats'
  %58  = Call(Callee: 'decode_layer', %55, %56, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0133', Output: 'layer_13_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0134', Output: 'layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0135', Output: 'layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0136', Output: 'layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0137', Output: 'layer_13_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0138', Output: 'layer_13_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0139', Output: 'layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0140', Output: 'layer_13_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0141', Output: 'layer_13_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0142', Output: 'layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 13), const(bool[] : False), %3, %4, %57): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_13_decode_layer_call'
  %59  = GetItem(%58, 0): // f32<4>[1,512] name='layer_13_decoder_output.boundary_packed'
  %60  = GetItem(%58, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_13_updated_state'
  %61  = GetItem(%58, 2): // f32[1,1,1] name='layer_14_decode_layer_call.norm_stats'
  %62  = Call(Callee: 'decode_layer', %59, %60, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0143', Output: 'layer_14_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0144', Output: 'layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0145', Output: 'layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0146', Output: 'layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0147', Output: 'layer_14_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0148', Output: 'layer_14_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0149', Output: 'layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0150', Output: 'layer_14_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0151', Output: 'layer_14_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0152', Output: 'layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 14), const(bool[] : False), %3, %4, %61): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_14_decode_layer_call'
  %63  = GetItem(%62, 0): // f32<4>[1,512] name='layer_14_decoder_output.boundary_packed'
  %64  = GetItem(%62, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_14_updated_state'
  %65  = GetItem(%62, 2): // f32[1,1,1] name='layer_15_decode_layer_call.norm_stats'
  %66  = Call(Callee: 'decode_layer', %63, %64, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0153', Output: 'layer_15_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0154', Output: 'layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0155', Output: 'layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0156', Output: 'layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0157', Output: 'layer_15_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0158', Output: 'layer_15_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0159', Output: 'layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0160', Output: 'layer_15_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0161', Output: 'layer_15_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0162', Output: 'layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 15), const(bool[] : False), %3, %4, %65): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_15_decode_layer_call'
  %67  = GetItem(%66, 0): // f32<4>[1,512] name='layer_15_decoder_output.boundary_packed'
  %68  = GetItem(%66, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_15_updated_state'
  %69  = GetItem(%66, 2): // f32[1,1,1] name='layer_16_decode_layer_call.norm_stats'
  %70  = Call(Callee: 'decode_layer', %67, %68, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0163', Output: 'layer_16_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0164', Output: 'layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0165', Output: 'layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0166', Output: 'layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0167', Output: 'layer_16_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0168', Output: 'layer_16_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0169', Output: 'layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0170', Output: 'layer_16_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0171', Output: 'layer_16_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0172', Output: 'layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 16), const(bool[] : False), %3, %4, %69): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_16_decode_layer_call'
  %71  = GetItem(%70, 0): // f32<4>[1,512] name='layer_16_decoder_output.boundary_packed'
  %72  = GetItem(%70, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_16_updated_state'
  %73  = GetItem(%70, 2): // f32[1,1,1] name='layer_17_decode_layer_call.norm_stats'
  %74  = Call(Callee: 'decode_layer', %71, %72, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0173', Output: 'layer_17_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0174', Output: 'layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0175', Output: 'layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0176', Output: 'layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0177', Output: 'layer_17_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0178', Output: 'layer_17_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0179', Output: 'layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0180', Output: 'layer_17_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0181', Output: 'layer_17_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0182', Output: 'layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 17), const(bool[] : False), %3, %4, %73): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_17_decode_layer_call'
  %75  = GetItem(%74, 0): // f32<4>[1,512] name='layer_17_decoder_output.boundary_packed'
  %76  = GetItem(%74, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_17_updated_state'
  %77  = GetItem(%74, 2): // f32[1,1,1] name='layer_18_decode_layer_call.norm_stats'
  %78  = Call(Callee: 'decode_layer', %75, %76, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0183', Output: 'layer_18_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0184', Output: 'layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0185', Output: 'layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0186', Output: 'layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0187', Output: 'layer_18_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0188', Output: 'layer_18_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0189', Output: 'layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0190', Output: 'layer_18_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0191', Output: 'layer_18_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0192', Output: 'layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 18), const(bool[] : False), %3, %4, %77): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_18_decode_layer_call'
  %79  = GetItem(%78, 0): // f32<4>[1,512] name='layer_18_decoder_output.boundary_packed'
  %80  = GetItem(%78, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_18_updated_state'
  %81  = GetItem(%78, 2): // f32[1,1,1] name='layer_19_decode_layer_call.norm_stats'
  %82  = Call(Callee: 'decode_layer', %79, %80, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0193', Output: 'layer_19_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0194', Output: 'layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0195', Output: 'layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0196', Output: 'layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0197', Output: 'layer_19_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0198', Output: 'layer_19_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0199', Output: 'layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0200', Output: 'layer_19_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0201', Output: 'layer_19_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0202', Output: 'layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 19), const(bool[] : False), %3, %4, %81): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_19_decode_layer_call'
  %83  = GetItem(%82, 0): // f32<4>[1,512] name='layer_19_decoder_output.boundary_packed'
  %84  = GetItem(%82, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_19_updated_state'
  %85  = GetItem(%82, 2): // f32[1,1,1] name='layer_20_decode_layer_call.norm_stats'
  %86  = Call(Callee: 'decode_layer', %83, %84, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0203', Output: 'layer_20_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0204', Output: 'layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0205', Output: 'layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0206', Output: 'layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0207', Output: 'layer_20_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0208', Output: 'layer_20_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0209', Output: 'layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0210', Output: 'layer_20_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0211', Output: 'layer_20_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0212', Output: 'layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 20), const(bool[] : False), %3, %4, %85): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_20_decode_layer_call'
  %87  = GetItem(%86, 0): // f32<4>[1,512] name='layer_20_decoder_output.boundary_packed'
  %88  = GetItem(%86, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_20_updated_state'
  %89  = GetItem(%86, 2): // f32[1,1,1] name='layer_21_decode_layer_call.norm_stats'
  %90  = Call(Callee: 'decode_layer', %87, %88, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0213', Output: 'layer_21_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0214', Output: 'layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0215', Output: 'layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0216', Output: 'layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0217', Output: 'layer_21_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0218', Output: 'layer_21_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0219', Output: 'layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0220', Output: 'layer_21_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0221', Output: 'layer_21_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0222', Output: 'layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 21), const(bool[] : False), %3, %4, %89): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_21_decode_layer_call'
  %91  = GetItem(%90, 0): // f32<4>[1,512] name='layer_21_decoder_output.boundary_packed'
  %92  = GetItem(%90, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_21_updated_state'
  %93  = GetItem(%90, 2): // f32[1,1,1] name='layer_22_decode_layer_call.norm_stats'
  %94  = Call(Callee: 'decode_layer', %91, %92, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0223', Output: 'layer_22_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0224', Output: 'layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0225', Output: 'layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0226', Output: 'layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0227', Output: 'layer_22_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0228', Output: 'layer_22_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0229', Output: 'layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0230', Output: 'layer_22_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0231', Output: 'layer_22_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0232', Output: 'layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 22), const(bool[] : False), %3, %4, %93): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_22_decode_layer_call'
  %95  = GetItem(%94, 0): // f32<4>[1,512] name='layer_22_decoder_output.boundary_packed'
  %96  = GetItem(%94, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_22_updated_state'
  %97  = GetItem(%94, 2): // f32[1,1,1] name='layer_23_decode_layer_call.norm_stats'
  %98  = Call(Callee: 'decode_layer', %95, %96, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0233', Output: 'layer_23_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0234', Output: 'layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0235', Output: 'layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0236', Output: 'layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0237', Output: 'layer_23_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0238', Output: 'layer_23_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0239', Output: 'layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0240', Output: 'layer_23_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0241', Output: 'layer_23_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0242', Output: 'layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 23), const(bool[] : False), %3, %4, %97): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_23_decode_layer_call'
  %99  = GetItem(%98, 0): // f32<4>[1,512] name='layer_23_decoder_output.boundary_packed'
  %100 = GetItem(%98, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_23_updated_state'
  %101 = GetItem(%98, 2): // f32[1,1,1] name='layer_24_decode_layer_call.norm_stats'
  %102 = Call(Callee: 'decode_layer', %99, %100, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0243', Output: 'layer_24_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0244', Output: 'layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0245', Output: 'layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0246', Output: 'layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0247', Output: 'layer_24_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0248', Output: 'layer_24_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0249', Output: 'layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0250', Output: 'layer_24_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0251', Output: 'layer_24_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0252', Output: 'layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 24), const(bool[] : False), %3, %4, %101): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_24_decode_layer_call'
  %103 = GetItem(%102, 0): // f32<4>[1,512] name='layer_24_decoder_output.boundary_packed'
  %104 = GetItem(%102, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_24_updated_state'
  %105 = GetItem(%102, 2): // f32[1,1,1] name='layer_25_decode_layer_call.norm_stats'
  %106 = Call(Callee: 'decode_layer', %103, %104, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0253', Output: 'layer_25_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0254', Output: 'layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0255', Output: 'layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0256', Output: 'layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0257', Output: 'layer_25_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0258', Output: 'layer_25_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0259', Output: 'layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0260', Output: 'layer_25_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0261', Output: 'layer_25_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0262', Output: 'layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 25), const(bool[] : False), %3, %4, %105): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_25_decode_layer_call'
  %107 = GetItem(%106, 0): // f32<4>[1,512] name='layer_25_decoder_output.boundary_packed'
  %108 = GetItem(%106, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_25_updated_state'
  %109 = GetItem(%106, 2): // f32[1,1,1] name='layer_26_decode_layer_call.norm_stats'
  %110 = Call(Callee: 'decode_layer', %107, %108, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0263', Output: 'layer_26_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0264', Output: 'layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0265', Output: 'layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0266', Output: 'layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0267', Output: 'layer_26_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0268', Output: 'layer_26_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0269', Output: 'layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0270', Output: 'layer_26_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0271', Output: 'layer_26_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0272', Output: 'layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 26), const(bool[] : False), %3, %4, %109): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_26_decode_layer_call'
  %111 = GetItem(%110, 0): // f32<4>[1,512] name='layer_26_decoder_output.boundary_packed'
  %112 = GetItem(%110, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_26_updated_state'
  %113 = GetItem(%110, 2): // f32[1,1,1] name='layer_27_decode_layer_call.norm_stats'
  %114 = Call(Callee: 'decode_layer', %111, %112, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0273', Output: 'layer_27_decode_layer_call.arg2.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0274', Output: 'layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0275', Output: 'layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'), ConstAssetRef(bf16<8,2,8>[128,128], Recipe: 'constant_recipe_0276', Output: 'layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0277', Output: 'layer_27_decode_layer_call.arg6.boundary_pack'), ConstAssetRef(bf16<8>[16], Recipe: 'constant_recipe_0278', Output: 'layer_27_decode_layer_call.arg7.boundary_pack'), ConstAssetRef(bf16<8,2,8>[128,256], Recipe: 'constant_recipe_0279', Output: 'layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0280', Output: 'layer_27_decode_layer_call.arg9.boundary_pack'), ConstAssetRef(bf16[2048,12288], Recipe: 'constant_recipe_0281', Output: 'layer_27_decode_layer_call.gu_weight_rowmajor'), ConstAssetRef(bf16<8,2,8>[384,256], Recipe: 'constant_recipe_0282', Output: 'layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'), const(i32[] : 27), const(bool[] : True), %3, %4, %113): // (f32<4>[1,512], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, f32[1,1,1]) !read_write<paged_attention_kv_cache> name='layer_27_decode_layer_call'
  %115 = GetItem(%114, 0): // f32<4>[1,512] name='layer_27_decoder_output.boundary_packed'
  %116 = GetItem(%114, 2): // f32[1,1,1] name='final_norm.stats'
  %117 = NN.NormApply(Axis: -1, Epsilon: 1e-06, OutputDtype: 'bfloat16', RoundBeforeScale: False, UseMean: False, %115, %116, ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0283', Output: 'final_norm.vectorized.pack2'), ConstAssetRef(bf16<4>[512], Recipe: 'constant_recipe_0001', Output: 'decode_layer_input_norm.vectorized.pack3')): // bf16<4>[1,512] name='final_norm.vectorized.compute'
  %118 = Tensors.Bitcast(Dtype: 'bfloat16', %117): // bf16[1,2048] name='final_norm'
  %119 = NTT.PackedMatMul(FusedReduce: False, OutputDataType: 'float32', RhsLayout: 'k_major', %118, ConstAssetRef(bf16<8,2,8>[128,18992], Recipe: 'constant_recipe_0000', Output: 'logits.vectorized.compute.rhs_k_major.physical'), %w0, %w0): // f32<2,4>[1,18992] name='logits.vectorized.compute'
  %120 = Tensors.Bitcast(Dtype: 'float32', %119): // f32[1,151936] name='logits'
  %121 = NN.GreedySample(%120): // i32[1] name='next_token'
  %122 = GetItem(%114, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='updated_state'
  return (%120, %121, %122)
}

// selections
select vectorization.final_norm = vectorization.norm_apply.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.logits = vectorization.matmul.n // kind=vectorization, origin=default-policy
select packing.lm_head = packing.k_major_n8_k16 // kind=packing, origin=default-policy
