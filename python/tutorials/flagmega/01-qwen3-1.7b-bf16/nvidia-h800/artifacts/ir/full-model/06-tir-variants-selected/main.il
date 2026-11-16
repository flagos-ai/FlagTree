// FlagMega high_level stage=selected_tir_variants semantic_hash=2fd1f683c2017b8f9ea58b7b9db4fda76423471da0133a0aec444d0a6bdb03f9
%main = fn(%input_ids: i32[1], %paged_attention_kv_cache: &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}): // (i32[1], &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}) -> ({f32[1,151936], (B,B), [1,151936], }, {i32[1], (B), [1], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]})
{
  weights {
    // display grouping only; not an execution schedule
    constant_recipe constant_recipe_0000 // fingerprint=7ff6b5a5c7bb83a512ede7e7bf834450fbf3a95cfdd32efbd71a7868eb2d776b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [4], WeightRef(bf16[151936,2048], 'embed_tokens.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.embed_tokens.weight')): // bf16<4>[151936,512] name='token_embedding.packed_weight'
      %1 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[151936,2048], 'embed_tokens.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.embed_tokens.weight')): // bf16<8>[151936,256] name='logits.vectorized.compute.rhs_k_major.k_vector'
      %2 = Tensors.Pack(Axes: [1], Lanes: [2], %1): // bf16<2,8>[151936,128] name='logits.vectorized.compute.rhs_k_major.k_pack'
      %3 = Tensors.Pack(Axes: [0], Lanes: [8], %2): // bf16<8,2,8>[18992,128] name='logits.vectorized.compute.rhs_k_major.n_vector'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,18992] name='logits.vectorized.compute.rhs_k_major.physical'
      %5 = Distributed.ShardedView(NewType: bf16<4>[151936,512], (B,S(C(4)@[0,1])), [y:8,x:16], Partial: , %0): // {bf16<4>[151936,512], (B,S(C(4)@[0,1])), [151936,4@y@x], } name='token_embedding.packed_weight.reshard16'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,18992], (B,S(BC(1)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,18992], (B,S(BC(1)@[0,1])), [128,⌈18992/128⌉@y@x], } name='logits.vectorized.compute.rhs_k_major.physical.reshard314'
      yield (%5, %6)
    }
    constant_recipe constant_recipe_0001 // fingerprint=cbde7c9c1a4a6e7cc1ed09517dc8a6516902c8e1103c8b48a0051788cb5e3db9
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], const(bf16[2048] : splat(0.0))): // bf16<4>[512] name='decode_layer_input_norm.vectorized.pack3'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='decode_layer_input_norm.vectorized.pack3.reshard21'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='decode_layer_input_norm.vectorized.pack3.reshard312'
      yield (%1, %2)
    }
    constant_recipe constant_recipe_0003 // fingerprint=d66c64d2b29923ff26165d0b82865bb203a06e395a4913baa838616ff6e48103
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.0.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.input_layernorm.weight')): // bf16<4>[512] name='layer_0_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_0_decode_layer_call.arg2.boundary_pack.reshard30'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_0_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0004 // fingerprint=5df882825615abdc3fc7d342f6b49508adfca13b878d95725d7c2371550f2ff7
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.0.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard31'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_0_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0005 // fingerprint=4d19b48d9952aa51e0cfc2b42c51ae63aa609dc9a8f4f181c197f1ccbeab2c3c
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.0.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard32'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_0_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0006 // fingerprint=4f2efc284a1776b5065dedcfdbc8f02d2b3afc9e7d8b25a33ea0f33a6b066d30
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.0.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard33'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_0_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0007 // fingerprint=c696a7844c662a49e246424aac63aac2f299d053bdd5376082c731698b77f530
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.0.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_0_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_0_decode_layer_call.arg6.boundary_pack.reshard34'
      yield (%1)
    }
    constant_recipe constant_recipe_0008 // fingerprint=f1b6231c20be3bdb2813ca6b4d70c4e4b28194a388c5cb25ee0ad871c9166cbb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.0.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_0_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_0_decode_layer_call.arg7.boundary_pack.reshard35'
      yield (%1)
    }
    constant_recipe constant_recipe_0009 // fingerprint=9ddb4b6f6e01c16d0ab77b30c4960cd63837c1586857f8c90d6212f391d2581d
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.0.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard36'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_0_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0010 // fingerprint=bcbc36fe61ea9f0ff8e418baf361d41b794d30bf481b781cf342a759164809ab
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.0.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_0_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_0_decode_layer_call.arg9.boundary_pack.reshard37'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_0_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0011 // fingerprint=7f432e7446111b7f88635cacd519b646ada15f18c8630c7529413d822efe7009
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.0.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard39'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_0_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0012 // fingerprint=a799feaeb5fe2a74cad677bbde879888d1a9b89a4c023b08510d4b698c1ad8f0
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.0.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.0.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_0_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_0_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_0_decode_layer_call.gu_weight_rowmajor.reshard38'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_0_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0013 // fingerprint=eed40ce9818f2959f06a4061864fc422507047b255ef6a0b5a3919dd0dc0b343
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.1.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.input_layernorm.weight')): // bf16<4>[512] name='layer_1_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_1_decode_layer_call.arg2.boundary_pack.reshard40'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_1_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0014 // fingerprint=ac5698412f38879b819049c7932bd76c21a26887a7e06c19a94a473b6b79ea10
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.1.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard41'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_1_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0015 // fingerprint=49d308d3533722d71ced080a809516993394f3fd4e3f9b1534d6785843e4cb9b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.1.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard42'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_1_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0016 // fingerprint=baa39fed9135529934eb60b69a4654fdd78b146a737d6b8bad8314fdfc71b5e8
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.1.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard43'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_1_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0017 // fingerprint=3a44e9a51d06633db2298b1d82845e9bfbd2fc4c3e705c57b3a370cbc2759525
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.1.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_1_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_1_decode_layer_call.arg6.boundary_pack.reshard44'
      yield (%1)
    }
    constant_recipe constant_recipe_0018 // fingerprint=60bb1c1f1bc6e5ab7f8ded53d1c36952006bccea91e3b517098759432e7cb12c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.1.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_1_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_1_decode_layer_call.arg7.boundary_pack.reshard45'
      yield (%1)
    }
    constant_recipe constant_recipe_0019 // fingerprint=8d887e289e070ac57b139e1a03317d941543088a6a32b0b7a71b954439b37c6f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.1.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard46'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_1_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0020 // fingerprint=759fd7c1bd324769375a106b5d0b739a9c297bda5aefe2e20cb355085aa716da
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.1.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_1_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_1_decode_layer_call.arg9.boundary_pack.reshard47'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_1_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0021 // fingerprint=5cd0c09c2eb2d65b7bff78e99adf239a647235beac4426d4e93166a82f9854b3
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.1.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard49'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_1_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0022 // fingerprint=3b3be02493ac287d7d36380f5785996ae7d2cc716e8fa9fd44c36cdacfb36283
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.1.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.1.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_1_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_1_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_1_decode_layer_call.gu_weight_rowmajor.reshard48'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_1_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0023 // fingerprint=079a3c7be0b0019d72592c346d14a5b7270635e87fc60ea4c31daa20dfab3fbd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.2.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.input_layernorm.weight')): // bf16<4>[512] name='layer_2_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_2_decode_layer_call.arg2.boundary_pack.reshard50'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_2_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0024 // fingerprint=e20523c7c0e0e332c4d05364527cfa0a9c339430e57b17bf5825ecb76ca31667
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.2.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard51'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_2_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0025 // fingerprint=28feb7975334b56434cf13f1b2ab3fb3e6a9058403f3cd8e562373087833f336
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.2.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard52'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_2_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0026 // fingerprint=f660a2b8c13e792ab9b318b9d689f618c8684dae48aa3d689ed2cf9bebdb6e7f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.2.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard53'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_2_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0027 // fingerprint=319a8f036df20acaf57e160eac7a31044c3e7df7300a70050c082b332f88a3b3
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.2.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_2_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_2_decode_layer_call.arg6.boundary_pack.reshard54'
      yield (%1)
    }
    constant_recipe constant_recipe_0028 // fingerprint=20915c6ffdc447c9d77e3fdff2cb5982302c18603238df667a96452bca702a1c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.2.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_2_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_2_decode_layer_call.arg7.boundary_pack.reshard55'
      yield (%1)
    }
    constant_recipe constant_recipe_0029 // fingerprint=4c9b8e34b5e73fb46e0356a03649c3d56e518252abd795277aa66574247b3162
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.2.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard56'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_2_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0030 // fingerprint=cfc5bf57cc32711244c1de290ebee98fa816e06e973c628077bd8ded18d91260
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.2.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_2_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_2_decode_layer_call.arg9.boundary_pack.reshard57'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_2_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0031 // fingerprint=974bfa183c5ea948c936a776d3260ab80ae7e7b0f603a722935836f5802e15f7
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.2.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard59'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_2_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0032 // fingerprint=b7bb0c3efb7dbd0caea208999fcd720f28c47dbe1568a6964fc4e1ea4fcba82b
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.2.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.2.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_2_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_2_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_2_decode_layer_call.gu_weight_rowmajor.reshard58'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_2_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0033 // fingerprint=f49e4daeb5cbe1f8af0865c5f5d0defc847dada292a8e1f1c0306dfbb761b4dd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.3.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.input_layernorm.weight')): // bf16<4>[512] name='layer_3_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_3_decode_layer_call.arg2.boundary_pack.reshard60'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_3_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0034 // fingerprint=d1ce3f8c737daea8732a4d1d6832d7a7727537ca91725ddcc5826f7df169666b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.3.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard61'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_3_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0035 // fingerprint=4d35da69e62024cb08e950d44615dc8ea028400b264234e4cfd21085e88b3cc0
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.3.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard62'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_3_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0036 // fingerprint=deeb3966f03c056f38effdc3c078b63488ab11d9b637417079e9698648999380
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.3.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard63'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_3_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0037 // fingerprint=3a1b42604dc83f0c701de6701ea666935d841f0c93b8eb15d654f0d96c030de9
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.3.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_3_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_3_decode_layer_call.arg6.boundary_pack.reshard64'
      yield (%1)
    }
    constant_recipe constant_recipe_0038 // fingerprint=b6d2618dce28c3bd141611ff6862756e8e1caee93cacf1158870d0f35a8eee73
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.3.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_3_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_3_decode_layer_call.arg7.boundary_pack.reshard65'
      yield (%1)
    }
    constant_recipe constant_recipe_0039 // fingerprint=6f7d25fb98065cd089aec7e804e7c21fc45bab2cf07aa615f212e8f1dcc8410e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.3.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard66'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_3_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0040 // fingerprint=5a1f4d9b46379cfa3291ef75b46d0a47622fd4d5bdfb0afa9da77db50acd7a5f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.3.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_3_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_3_decode_layer_call.arg9.boundary_pack.reshard67'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_3_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0041 // fingerprint=8a2be394a568e6ddfcdd7f3430ee750c64ac288ec748d4855219475575ecf113
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.3.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard69'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_3_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0042 // fingerprint=6d4220bc7b426b992bf4b81f4dd7812a42e2a6d06659f95c04cb5f16bc44e016
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.3.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.3.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_3_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_3_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_3_decode_layer_call.gu_weight_rowmajor.reshard68'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_3_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0043 // fingerprint=0adafb482c45374533e7eef98dc25cbeccd84efa1e7e60cdcb991cfed923e881
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.4.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.input_layernorm.weight')): // bf16<4>[512] name='layer_4_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_4_decode_layer_call.arg2.boundary_pack.reshard70'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_4_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0044 // fingerprint=1ca82275e146b8ca6ec3c36c580e6204f1cf3508d9ab2cca2756a9ca48bcf199
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.4.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard71'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_4_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0045 // fingerprint=49d0e50060e16ed3d3155178ffc2de91f5cc72622cde3680cfcbaaf9dbd556a9
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.4.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard72'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_4_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0046 // fingerprint=1ed71b974feb7cdf07305cf56b57f192e440f52de4d823c691d764ba7694607b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.4.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard73'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_4_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0047 // fingerprint=4fa9b6555eaf4488e768adc57fa99c079893d76ed14e872444e65d671c9d102a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.4.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_4_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_4_decode_layer_call.arg6.boundary_pack.reshard74'
      yield (%1)
    }
    constant_recipe constant_recipe_0048 // fingerprint=8fabd49cf2649b8c0ca0c9fd3b8609fe427a69eeea1593ea64daa10aa56c52a0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.4.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_4_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_4_decode_layer_call.arg7.boundary_pack.reshard75'
      yield (%1)
    }
    constant_recipe constant_recipe_0049 // fingerprint=25e55cbb37072856da68bcebeafab501b76b9b59733d6793d68ae7d28b212c58
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.4.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard76'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_4_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0050 // fingerprint=a71a5de518aa178ad22830e87dfec6bb3b65f6bb15ce7981219a932b79941f79
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.4.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_4_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_4_decode_layer_call.arg9.boundary_pack.reshard77'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_4_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0051 // fingerprint=4fb1b91de95c82a9165983b1d1f229cb34a73971f05b4a93dff628bfcb87cc74
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.4.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard79'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_4_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0052 // fingerprint=18bd45538b25fe396b2371ec17c1ca75891778ab2061aa602370bfdb40289abb
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.4.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.4.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_4_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_4_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_4_decode_layer_call.gu_weight_rowmajor.reshard78'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_4_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0053 // fingerprint=a220bda13daa24b99c06c250d028746070feaa6757b1e53ad6967bd9dc11508f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.5.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.input_layernorm.weight')): // bf16<4>[512] name='layer_5_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_5_decode_layer_call.arg2.boundary_pack.reshard80'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_5_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0054 // fingerprint=9fe110a512a25c002baa8c52fb089951e0d8043086cd831c32f00ac927bdbe45
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.5.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard81'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_5_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0055 // fingerprint=0c9dd536b52593d633332821834d5d47e49f7013c3a84bceb34bde32b698eea0
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.5.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard82'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_5_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0056 // fingerprint=70c54e2a27c1b3442c10b0ac7179f5894694573d50a718a4b635c42373b3c88c
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.5.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard83'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_5_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0057 // fingerprint=245385bb6126ae8e1b6f547cbf8d359d8979f34524c0fd48b1b205f68a173234
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.5.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_5_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_5_decode_layer_call.arg6.boundary_pack.reshard84'
      yield (%1)
    }
    constant_recipe constant_recipe_0058 // fingerprint=794d9db54bbe81feb677d0e280aeec8c7e94aa89f7fa281f3b97b1328fa54887
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.5.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_5_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_5_decode_layer_call.arg7.boundary_pack.reshard85'
      yield (%1)
    }
    constant_recipe constant_recipe_0059 // fingerprint=027d66e919a9fb2217dfc93138098974f7c5b3065a66253b55aac8c13f461e80
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.5.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard86'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_5_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0060 // fingerprint=ffffce01315fd3203d4298947a6e8f87a5ca5e8b1290d4b6160ce62d0a14261a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.5.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_5_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_5_decode_layer_call.arg9.boundary_pack.reshard87'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_5_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0061 // fingerprint=87c373fae203c7a8f4c75774d83ebd408e2b98a5a846357abc00cf5ffa3e18e9
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.5.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard89'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_5_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0062 // fingerprint=b02fd98537c4848e39470a7632b6d40b722b1013824fbeee9c25dc351d46e04c
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.5.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.5.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_5_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_5_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_5_decode_layer_call.gu_weight_rowmajor.reshard88'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_5_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0063 // fingerprint=d2424ac5e56aefb35b04a7378da9afbfd195010d08d968bdfbb537e8f15b4cf6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.6.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.input_layernorm.weight')): // bf16<4>[512] name='layer_6_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_6_decode_layer_call.arg2.boundary_pack.reshard90'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_6_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0064 // fingerprint=b64b6ed086457643fd7a6b39722ade3868d893df26ccfedfcd3dca3869e77638
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.6.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard91'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_6_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0065 // fingerprint=497d5f41f16d18960275c134d9cf47cb734b1f5c1c3cfe2847a94d77dc4c2074
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.6.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard92'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_6_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0066 // fingerprint=10271e6e416d46764d16dd047170024fcc054629d027755b6e3bee645932279e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.6.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard93'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_6_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0067 // fingerprint=70cc78c062c80298e3e4be09a5c1c44e94344f9c6700e8976f0ccd1512c163a1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.6.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_6_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_6_decode_layer_call.arg6.boundary_pack.reshard94'
      yield (%1)
    }
    constant_recipe constant_recipe_0068 // fingerprint=1065627638d13d49fd49d641108bc3f35735a0ac0ef5bf99ee5099fcefc9df41
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.6.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_6_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_6_decode_layer_call.arg7.boundary_pack.reshard95'
      yield (%1)
    }
    constant_recipe constant_recipe_0069 // fingerprint=c0a1185093ea1bd3446cdc8c62d6c82da7e1e3c91782a012246ea853bc00200f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.6.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard96'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_6_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0070 // fingerprint=574b9fef64290957d8251d3eac02fa4616a682bc539d734a5fa7946be407959f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.6.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_6_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_6_decode_layer_call.arg9.boundary_pack.reshard97'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_6_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0071 // fingerprint=a15346bb7928eba7889cfc9fc491cef4d134bb10b98946b6ef3efdaacfe14097
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.6.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard99'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_6_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0072 // fingerprint=f78b0a43aa44e1befb209791bf0dc5d2037cbc8da4ae8bd0f611830a2150f9fc
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.6.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.6.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_6_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_6_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_6_decode_layer_call.gu_weight_rowmajor.reshard98'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_6_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0073 // fingerprint=c7c1586be4bff68dbfdf8ecd301e96344e03d6eaec455c28fb593c1665a6f3f0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.7.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.input_layernorm.weight')): // bf16<4>[512] name='layer_7_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_7_decode_layer_call.arg2.boundary_pack.reshard100'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_7_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0074 // fingerprint=d5ceb60c2c3df5a983f36ed6c5feda3f32634ffe49b925d8b6e2a6c900a7ae7d
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.7.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard101'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_7_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0075 // fingerprint=f732b141ab3c98057e90f154f83f2583a9f2edfec7e9e1961d31be3de29bfd74
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.7.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard102'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_7_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0076 // fingerprint=30715dc38b3e7a55e41d1b2763d539117e038625aab0a1f5fe1eddaada436e53
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.7.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard103'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_7_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0077 // fingerprint=1f79b638c8d41dbcedaf9d403dbd77094cd27b04a01c40fe0bfcf6ed564a7bd0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.7.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_7_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_7_decode_layer_call.arg6.boundary_pack.reshard104'
      yield (%1)
    }
    constant_recipe constant_recipe_0078 // fingerprint=a62790f63f13e96770185a9615ef95924cbaab08210633d680dd1db9adcff966
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.7.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_7_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_7_decode_layer_call.arg7.boundary_pack.reshard105'
      yield (%1)
    }
    constant_recipe constant_recipe_0079 // fingerprint=45c621ad1d3717f15035719f5b48a3940e400c025d92e499502407582d6f7bee
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.7.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard106'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_7_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0080 // fingerprint=0bf6e7da527d023cddb5b76805391b9c1d182e12ae062009a13d81e62b6941c6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.7.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_7_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_7_decode_layer_call.arg9.boundary_pack.reshard107'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_7_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0081 // fingerprint=b8d7518f6e5c2fc3c55c89d03e99b4e72977b27001df9aa7b71d926aa731a707
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.7.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard109'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_7_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0082 // fingerprint=211bbb70edc688d1614ad0c3a91dea5a25d20f514a2aedf2ffde67a9324c3099
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.7.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.7.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_7_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_7_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_7_decode_layer_call.gu_weight_rowmajor.reshard108'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_7_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0083 // fingerprint=612c3c1267d172ef96e75fa9cb838b831ca0831d6eb79754acf4f43b36db7382
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.8.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.input_layernorm.weight')): // bf16<4>[512] name='layer_8_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_8_decode_layer_call.arg2.boundary_pack.reshard110'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_8_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0084 // fingerprint=8d40a01313f6952060d6856807979d5729947e3fa7626fa30e1cd3ddfca96cae
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.8.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard111'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_8_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0085 // fingerprint=6c0c6094faa7d88f88085947ca9ce1580a5445cc3cf991f5ce261582d28de408
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.8.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard112'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_8_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0086 // fingerprint=2afc09d8a1c168ae33cc6c30ab3bf597dbcb871ab45b9e492d7f1f86ef837869
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.8.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard113'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_8_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0087 // fingerprint=13a19944fb3580af8fcf99d562e7c8fa53a757a14f37076059748b4c459558af
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.8.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_8_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_8_decode_layer_call.arg6.boundary_pack.reshard114'
      yield (%1)
    }
    constant_recipe constant_recipe_0088 // fingerprint=f0d9e27a0d6bf3e30c254033c42b729420f045e7a168379bbc3771d6124c0c08
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.8.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_8_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_8_decode_layer_call.arg7.boundary_pack.reshard115'
      yield (%1)
    }
    constant_recipe constant_recipe_0089 // fingerprint=f9c53704403a4d5d43c857a02c40299ec1be7b5aceac9fb15b5476500d808885
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.8.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard116'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_8_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0090 // fingerprint=1205a41c7459347eac11e5f032dd32f16567ad5724a864d22077b2438eeabd74
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.8.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_8_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_8_decode_layer_call.arg9.boundary_pack.reshard117'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_8_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0091 // fingerprint=28f642fef5aed79eda588f0351d204ae148d925512b9d63f242e47691557a22a
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.8.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard119'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_8_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0092 // fingerprint=bcc6fac8142c86b47a9301b29b10ce4864d3b2fb6eabbf77e439bc4a98948199
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.8.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.8.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_8_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_8_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_8_decode_layer_call.gu_weight_rowmajor.reshard118'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_8_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0093 // fingerprint=7625b2b3ee32dc2e42a8bde773035ecc1ee20efa9638e9e3775c95f8dfcd8c4c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.9.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.input_layernorm.weight')): // bf16<4>[512] name='layer_9_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_9_decode_layer_call.arg2.boundary_pack.reshard120'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_9_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0094 // fingerprint=1360dfc1075699d41652e3847474f5113edfab76e2ef05d0d05dad5c111de5b6
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.9.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard121'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_9_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0095 // fingerprint=3e36fec0009ee1277b954939c922f0d44d1e734cb8465a75bd7e6f860f306dcf
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.9.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard122'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_9_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0096 // fingerprint=d17b0e10164c7cc0d1b84dd31a1454bd8ced1a136ef0b878d52b5f98f430bcdf
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.9.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard123'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_9_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0097 // fingerprint=1a1f7e95f002d7ce643332ca94d62aa2cce0c55f9196580cf5846138a2e38aa3
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.9.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_9_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_9_decode_layer_call.arg6.boundary_pack.reshard124'
      yield (%1)
    }
    constant_recipe constant_recipe_0098 // fingerprint=0e22fa9da070188c2b61b521ae420074cf882c553ba32999946ae02f36ab618a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.9.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_9_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_9_decode_layer_call.arg7.boundary_pack.reshard125'
      yield (%1)
    }
    constant_recipe constant_recipe_0099 // fingerprint=62337ab60d6944ee9f33aadcb15beb8490b14e0b44b5f80bc7e31bcbea942e88
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.9.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard126'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_9_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0100 // fingerprint=808f11813d9c56322a1f12a2279e50642ddf767a9279277602d276b0e877f696
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.9.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_9_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_9_decode_layer_call.arg9.boundary_pack.reshard127'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_9_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0101 // fingerprint=113d152c55d23066d09d23eba76805674070aa88b8a8b8539b2bb3d0482f322b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.9.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard129'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_9_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0102 // fingerprint=9b94aed24f1bed5ab88863455aed7ce212cdd9093662c7edc3d6c265685201d3
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.9.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.9.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_9_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_9_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_9_decode_layer_call.gu_weight_rowmajor.reshard128'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_9_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0103 // fingerprint=07604d37cd6d3ec8b69d811955ed54c9747db72f5095e576cda758f67f69b59d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.10.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.input_layernorm.weight')): // bf16<4>[512] name='layer_10_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_10_decode_layer_call.arg2.boundary_pack.reshard130'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_10_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0104 // fingerprint=5ffc9707611186790eb2b669d537fa929c577f989a55081da2fb9d6a62b0f5f6
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.10.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard131'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_10_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0105 // fingerprint=7af62c5534e5742697101493f2c56cf29a3eb735bf38cfb84c84c53203bf95c5
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.10.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard132'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_10_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0106 // fingerprint=2cb1b37dd54a08a07db8cd76a4dc3676cafe202d81d90c000a44a3ff336c9414
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.10.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard133'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_10_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0107 // fingerprint=bff0b22d2dcb73b95dc10eeb5842a559d90a687b700071095a1fc10e07884486
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.10.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_10_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_10_decode_layer_call.arg6.boundary_pack.reshard134'
      yield (%1)
    }
    constant_recipe constant_recipe_0108 // fingerprint=dbcaf65c4e2a7945387401f69d7ba0dc42670218ef1ead392c6bd8fc08f072db
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.10.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_10_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_10_decode_layer_call.arg7.boundary_pack.reshard135'
      yield (%1)
    }
    constant_recipe constant_recipe_0109 // fingerprint=003db388b71cecab77ec6485f6c7c72b2e70f84a23f0dc5c69fc80d0deaee734
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.10.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard136'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_10_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0110 // fingerprint=9af03443e3bbba12bec72780a5c12933dc2fa12fd005a04b5a3c49831b841799
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.10.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_10_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_10_decode_layer_call.arg9.boundary_pack.reshard137'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_10_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0111 // fingerprint=88c5192188ed0b5e3803b03b7539d9596fdcd0aced49e34c5b9b2ff5437baa3a
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.10.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard139'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_10_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0112 // fingerprint=310171febc938d0934e582202ae13f3f43314368a4444adf61128c848dc231b2
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.10.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.10.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_10_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_10_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_10_decode_layer_call.gu_weight_rowmajor.reshard138'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_10_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0113 // fingerprint=8e0b38e72bc91130285438350d10a8b6a1a46cc432967ad707c89b18a9862670
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.11.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.input_layernorm.weight')): // bf16<4>[512] name='layer_11_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_11_decode_layer_call.arg2.boundary_pack.reshard140'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_11_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0114 // fingerprint=09eda05ec3dbdb8f45f24a1be5646ad400a8c45c5bcb1af4df0b9959fe595cbf
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.11.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard141'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_11_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0115 // fingerprint=eda639f71581e6db71b88c785c75f1f4fa3ba9d9e68dcd42307892acee098699
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.11.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard142'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_11_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0116 // fingerprint=ce22a74b8a8c669ce6ab50b743f61146613bdfb755ec8a6ef44fd8f03f936d34
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.11.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard143'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_11_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0117 // fingerprint=c2e8f2ece4dc217ccce1457298f33ac99b3b73212bfeb02cdf3d423c0e5d088c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.11.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_11_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_11_decode_layer_call.arg6.boundary_pack.reshard144'
      yield (%1)
    }
    constant_recipe constant_recipe_0118 // fingerprint=a72e1762d476eae74d066c8718228f02ce9b9a7c50b6d39dd78b14e856869d97
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.11.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_11_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_11_decode_layer_call.arg7.boundary_pack.reshard145'
      yield (%1)
    }
    constant_recipe constant_recipe_0119 // fingerprint=56178126c426f59d3b29a426acdbe8daf376f8c8c55a49d9915d3f5d42f16842
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.11.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard146'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_11_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0120 // fingerprint=c283854f6a6c7a88792efc290d46ee10cd01093be23d03e35b09f72ccb83ab6b
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.11.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_11_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_11_decode_layer_call.arg9.boundary_pack.reshard147'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_11_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0121 // fingerprint=ac55b42c6a7a2964ba8762a1f9240c2915da3ac9271c2d9e7126abc57d2207c3
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.11.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard149'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_11_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0122 // fingerprint=eb94fcce7d254901d88722a637aac293f357ed9fe5e7e775a299550438e9cf2c
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.11.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.11.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_11_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_11_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_11_decode_layer_call.gu_weight_rowmajor.reshard148'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_11_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0123 // fingerprint=e79ce7aa023b5e97a0727cefe0b1715a4db32fcbfffdd0e1343f2738eeaeaaef
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.12.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.input_layernorm.weight')): // bf16<4>[512] name='layer_12_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_12_decode_layer_call.arg2.boundary_pack.reshard150'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_12_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0124 // fingerprint=374d8817c408b5b25b4d9d5181f1fbe0f72df0593914d477a6edc7e21c17174e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.12.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard151'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_12_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0125 // fingerprint=04daddd105a6bcde3a880b3e482e5da6f531ce6499b316521143f5b7f49938fd
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.12.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard152'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_12_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0126 // fingerprint=eba61cd8b57ce038fcd0b04267848e08a05314d7a1c60abb4a75128e5656c7ee
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.12.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard153'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_12_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0127 // fingerprint=2fd75ecf8f22bad1585fec4ed5d2e191366cc8b15b9b957eb6c2cd77f90eba60
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.12.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_12_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_12_decode_layer_call.arg6.boundary_pack.reshard154'
      yield (%1)
    }
    constant_recipe constant_recipe_0128 // fingerprint=e0440087fd7605e7c474b3c6c7c4d3eed98722d13ec393745a92a6e3ad425bb7
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.12.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_12_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_12_decode_layer_call.arg7.boundary_pack.reshard155'
      yield (%1)
    }
    constant_recipe constant_recipe_0129 // fingerprint=71080737c77efb17f64fe750960da0c195ed1198295c5a33e50e9c911d9d0f23
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.12.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard156'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_12_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0130 // fingerprint=beb7e70f1c0248b32661b708e6d5dc08804bb344f8219e1ed8320547cf9a1841
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.12.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_12_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_12_decode_layer_call.arg9.boundary_pack.reshard157'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_12_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0131 // fingerprint=acca4fb888df2ec91c12ffeec3b45a11ba7443c36be502e6ebfac3c736c641f2
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.12.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard159'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_12_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0132 // fingerprint=50d7ab8ecff0308c15d56453a94f0ab6b0f2f5ebdd76d3d4e335adea080cae2a
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.12.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.12.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_12_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_12_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_12_decode_layer_call.gu_weight_rowmajor.reshard158'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_12_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0133 // fingerprint=1ac21ec8eefd20d9eaa0e7b1e7f0b52d647a69afc26e8e4fb5895c0f492fe81a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.13.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.input_layernorm.weight')): // bf16<4>[512] name='layer_13_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_13_decode_layer_call.arg2.boundary_pack.reshard160'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_13_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0134 // fingerprint=4230c8f8860cbc7a0ae15845aa4708d20a68ee37a205dc03495a6f1abb3ec32f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.13.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard161'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_13_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0135 // fingerprint=b35a5029e730baf74ea06fa25c541b60de3dedbdfa86bb0a1a9e75a7db031a39
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.13.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard162'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_13_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0136 // fingerprint=04772f1a9eca3e00f3a4b27dc32bf77a12ebc05d4419511d935b04db51ee9815
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.13.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard163'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_13_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0137 // fingerprint=849e5f3c7919503190a888b6f4d492614ee8bc14fa5145e2156977c6d48fbec4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.13.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_13_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_13_decode_layer_call.arg6.boundary_pack.reshard164'
      yield (%1)
    }
    constant_recipe constant_recipe_0138 // fingerprint=e4ad538a1551a40266e0260b57b2c8bfb6a0d6230cc3a9229e71a36c7a1bbc16
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.13.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_13_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_13_decode_layer_call.arg7.boundary_pack.reshard165'
      yield (%1)
    }
    constant_recipe constant_recipe_0139 // fingerprint=98a1f475fa683549e9816833fd265db536dabac54867b04197f24f8f0596ec08
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.13.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard166'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_13_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0140 // fingerprint=9cdeab59bd6e4757022034d22371ad3f92f2c2908f188c17d28dd332dfe087c2
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.13.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_13_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_13_decode_layer_call.arg9.boundary_pack.reshard167'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_13_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0141 // fingerprint=e66153dd09f1234c789e83684efac5d0229ae4b4a2b3110b42c7af3b72733a14
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.13.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard169'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_13_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0142 // fingerprint=e972d9e7a208eb3eb66100f5a54f46bea79b9abce82770d704f9307d9242a89f
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.13.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.13.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_13_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_13_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_13_decode_layer_call.gu_weight_rowmajor.reshard168'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_13_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0143 // fingerprint=6824d1025a8bed8544cb623bfca24ebe85acdea978208688029f1e21f307c0a6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.14.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.input_layernorm.weight')): // bf16<4>[512] name='layer_14_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_14_decode_layer_call.arg2.boundary_pack.reshard170'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_14_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0144 // fingerprint=4ab66e420e0d2cd485dc9e1f2c7628c11ef46b7de7c5e498662dcad0679bb104
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.14.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard171'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_14_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0145 // fingerprint=a984f602f23d8d56b8e8f1fb4817f035f0bb8fb3ea957e58dbbef2e8f554f795
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.14.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard172'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_14_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0146 // fingerprint=16eff6a748c6c9efaabc5c36501a4b3f3e714275a040841b189a9d139bf04c47
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.14.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard173'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_14_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0147 // fingerprint=8a0d12d42666b4a8e01400ff9c5f22415c156da81d71e7e9ba731815cd42001f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.14.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_14_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_14_decode_layer_call.arg6.boundary_pack.reshard174'
      yield (%1)
    }
    constant_recipe constant_recipe_0148 // fingerprint=bbded5b8d950d696b09ef4dc1d0d7d237edabdd8dde9c0b1d3c3d73053faa55c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.14.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_14_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_14_decode_layer_call.arg7.boundary_pack.reshard175'
      yield (%1)
    }
    constant_recipe constant_recipe_0149 // fingerprint=e98edf1f69223cd47d0740d051aae2f215757a7d6cfb1734faf1b1adcc4c60b6
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.14.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard176'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_14_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0150 // fingerprint=92ce87e0d1981755b29125143f2d63573902063d2b0384b71aa6be8734c32a7c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.14.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_14_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_14_decode_layer_call.arg9.boundary_pack.reshard177'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_14_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0151 // fingerprint=9cbd401d9d334216f072c3606a540a908e963335a9567d880d77eaf365910c98
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.14.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard179'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_14_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0152 // fingerprint=f56e14c6cad798720ec2de2d7d5bde3c933d73ad67a42da0343b1b2712af6d92
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.14.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.14.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_14_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_14_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_14_decode_layer_call.gu_weight_rowmajor.reshard178'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_14_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0153 // fingerprint=e8dab58e1d9b47be39fc586be12f3c492e40694f8c3ee08181d1b86959bf1a59
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.15.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.input_layernorm.weight')): // bf16<4>[512] name='layer_15_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_15_decode_layer_call.arg2.boundary_pack.reshard180'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_15_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0154 // fingerprint=d3ce31eeb150258fd7f05a4f483706799cd22ffdf2048e86da7f85f90938bf17
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.15.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard181'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_15_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0155 // fingerprint=84acf17e8d64091bc5b0aebfa8851ed91c0f3036512dd160e61ee43d4c5d2168
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.15.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard182'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_15_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0156 // fingerprint=02a72d540609515a5c351a226b821c925ea9e06892c2cdfc523b504281afab80
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.15.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard183'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_15_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0157 // fingerprint=36e9b2c2ca3a36f86875548410081d48c90016fb372efbf9c94fcea2218d435d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.15.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_15_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_15_decode_layer_call.arg6.boundary_pack.reshard184'
      yield (%1)
    }
    constant_recipe constant_recipe_0158 // fingerprint=7dc2a6b5b13b077ab29b49d2cda9e87d5615fa40cd3a175b224ba430a8eba870
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.15.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_15_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_15_decode_layer_call.arg7.boundary_pack.reshard185'
      yield (%1)
    }
    constant_recipe constant_recipe_0159 // fingerprint=5e6fe3764b4f9a118ab43ca0de7a7547cd73c6d62c36f582d87489c13b0796dc
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.15.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard186'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_15_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0160 // fingerprint=334a1c1854f0d9f486f6fc8a873a4094e06c4139b1d7ad6edececf97555a473d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.15.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_15_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_15_decode_layer_call.arg9.boundary_pack.reshard187'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_15_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0161 // fingerprint=9c6c3810e8612f30c08f750909797aa49cd4bc3786b6de6eab830ddd56a3a8ce
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.15.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard189'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_15_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0162 // fingerprint=2e44cbca09d7f7e48856c50f1ff6a18d0455425267d7c4bfc02cdaf4f98c2fc8
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.15.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.15.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_15_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_15_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_15_decode_layer_call.gu_weight_rowmajor.reshard188'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_15_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0163 // fingerprint=e44f12a2ba10565374e8280aa7ae68c5b468dc9c5e4db2ebe676401d3b2711b4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.16.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.input_layernorm.weight')): // bf16<4>[512] name='layer_16_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_16_decode_layer_call.arg2.boundary_pack.reshard190'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_16_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0164 // fingerprint=9aee001fdb7a6af564f939fa81551f62d0f5cdd86a1cad6d75ce5d17fbb40193
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.16.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard191'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_16_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0165 // fingerprint=ac1f909c395dbbded360fe793dd7174f923f2f9fa2775a092b91ca365302792e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.16.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard192'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_16_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0166 // fingerprint=1da15d4def2c792bf05bc09b4c4c56d1d92b2bf4173c7c077e4aa44e0f11590b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.16.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard193'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_16_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0167 // fingerprint=d167dd102f4cfc494f39aa328e002431b1f299476d813b37e7ec0fe75189e103
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.16.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_16_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_16_decode_layer_call.arg6.boundary_pack.reshard194'
      yield (%1)
    }
    constant_recipe constant_recipe_0168 // fingerprint=527ade5970b4a4270b201580c026f1085eed45f6ab633c5535a1e5e47ec601eb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.16.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_16_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_16_decode_layer_call.arg7.boundary_pack.reshard195'
      yield (%1)
    }
    constant_recipe constant_recipe_0169 // fingerprint=ef04dc48fd16b23a08116aaa12e24bf7b2d6f52abdac5e8c7698263851eefffd
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.16.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard196'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_16_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0170 // fingerprint=bc626ffe4fbb6c87cf91c13e5ff32b26334b1d1453aec69b7aa4f73e5d3cd129
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.16.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_16_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_16_decode_layer_call.arg9.boundary_pack.reshard197'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_16_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0171 // fingerprint=9b6a2d57369478f364334ed4a7b07d4cca7bb1929221d14749f4287967de676c
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.16.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard199'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_16_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0172 // fingerprint=9c210631c65b7293ed19b23e5f9cde10023c6acbcd8d9b66eee2318fa6691739
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.16.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.16.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_16_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_16_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_16_decode_layer_call.gu_weight_rowmajor.reshard198'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_16_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0173 // fingerprint=659c4eac1f9976db6f0c11095ee837e18c3d215a7552bda63b8338e0b527b99d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.17.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.input_layernorm.weight')): // bf16<4>[512] name='layer_17_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_17_decode_layer_call.arg2.boundary_pack.reshard200'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_17_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0174 // fingerprint=170ab0b9d855d4ecf85966bf161285db035846be4a714826d572f41837e52336
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.17.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard201'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_17_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0175 // fingerprint=0e407c7e2a6dd4560cfce103c43d7b815644fc73488312103062aa1116e0b3bb
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.17.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard202'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_17_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0176 // fingerprint=9fa7a8ae6601bcbb9f6e328394a1d5d29f44d97717bb02d89f4db205ebf61935
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.17.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard203'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_17_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0177 // fingerprint=783863b6ea181b4c2428811ed774a448dfb3098cfb20132dea75a245121d5db6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.17.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_17_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_17_decode_layer_call.arg6.boundary_pack.reshard204'
      yield (%1)
    }
    constant_recipe constant_recipe_0178 // fingerprint=983dd7ca682bc7f0980892a66a85e49d69fc173fb0126ddb9c8e2a1e61e18580
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.17.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_17_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_17_decode_layer_call.arg7.boundary_pack.reshard205'
      yield (%1)
    }
    constant_recipe constant_recipe_0179 // fingerprint=16120e0fa7759008eeedb5facb31e96356da5647ac274f6c83ec077857dc996f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.17.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard206'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_17_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0180 // fingerprint=e9f94953b3ad033e2ce7f4a2c9fe809cdbc82843df42ec93ea6acd10e0c536ad
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.17.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_17_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_17_decode_layer_call.arg9.boundary_pack.reshard207'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_17_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0181 // fingerprint=b952838b29110656721f39dd93ffb13394611ddac74cad1978044269e4ca7576
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.17.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard209'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_17_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0182 // fingerprint=d3baf632e19248f203e222059e3a7c63ac307db00c5bfe552dce10a031c8c3ed
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.17.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.17.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_17_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_17_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_17_decode_layer_call.gu_weight_rowmajor.reshard208'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_17_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0183 // fingerprint=9272cc0106457d667b56c50b6f134b37d299ee4f92a127c0484e0dfbd3c7febb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.18.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.input_layernorm.weight')): // bf16<4>[512] name='layer_18_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_18_decode_layer_call.arg2.boundary_pack.reshard210'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_18_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0184 // fingerprint=b29dafbc2d4067d3b6bff765f9e3b364c93359e9644aedb9bcc0d0e6cb450d5c
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.18.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard211'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_18_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0185 // fingerprint=98d5a1844c746c0a5eb71ca2513abc1a2ece3ef9787ec52b03c56261a2e47ba9
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.18.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard212'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_18_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0186 // fingerprint=d040d5ec6bd7f7e9e1025fbc3181750dd8d215289f8359034f13be36182d5c31
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.18.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard213'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_18_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0187 // fingerprint=d88e47ee9af433d8d2359ffc7d7e74d62f84efae37e4b5591a5a958af3e7e5ea
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.18.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_18_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_18_decode_layer_call.arg6.boundary_pack.reshard214'
      yield (%1)
    }
    constant_recipe constant_recipe_0188 // fingerprint=ed48bdd6b9b74430b03699a9cb746bcf0c5fd12f8ab89b0bc239eb62bf0339fd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.18.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_18_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_18_decode_layer_call.arg7.boundary_pack.reshard215'
      yield (%1)
    }
    constant_recipe constant_recipe_0189 // fingerprint=d4813a1423bff564169b1fc22943bbd32fa9c11676e2bcd3e0f921b6a9c01644
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.18.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard216'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_18_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0190 // fingerprint=c625ae5c105236fc715a375c0f8024107e775834e23a4e2934f67e85ec040bda
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.18.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_18_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_18_decode_layer_call.arg9.boundary_pack.reshard217'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_18_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0191 // fingerprint=58747f5597ac6b48d1d1f3736396b7530ca08add834b3e9a7db47f4a318ccf03
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.18.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard219'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_18_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0192 // fingerprint=16e791aa0b072d538b11537b579c46132e323888afc7132b00706acdd9cb1248
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.18.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.18.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_18_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_18_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_18_decode_layer_call.gu_weight_rowmajor.reshard218'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_18_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0193 // fingerprint=125e01748666437caa5f08f9a5259575bc6575a61f756f3a6f9b6572691b42c1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.19.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.input_layernorm.weight')): // bf16<4>[512] name='layer_19_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_19_decode_layer_call.arg2.boundary_pack.reshard220'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_19_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0194 // fingerprint=0f593e4c6052eab12bdc70efbfa031fb3491c797f33dd2275c4d68f12505ab54
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.19.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard221'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_19_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0195 // fingerprint=7b6c8bee653dcec2172ec89b2b576625ead90fc523667402c5dacc755001fc8b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.19.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard222'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_19_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0196 // fingerprint=b17d121e389dc36fa837065b9e99477aec7394f63854e8e19048ab74a93dba34
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.19.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard223'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_19_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0197 // fingerprint=26439b80677bbbaceeec831a46ec5f63b0fa1b82e6a23216cd4afa80945d1fb4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.19.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_19_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_19_decode_layer_call.arg6.boundary_pack.reshard224'
      yield (%1)
    }
    constant_recipe constant_recipe_0198 // fingerprint=26b4f5793ad8343f5a065949924ccb9799f4321f2c59f9ed4455775529e45a5f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.19.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_19_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_19_decode_layer_call.arg7.boundary_pack.reshard225'
      yield (%1)
    }
    constant_recipe constant_recipe_0199 // fingerprint=40700cc217f3e89e1626ffb3d260ed8c6d92e4c0d38ce720b9273039e624adee
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.19.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard226'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_19_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0200 // fingerprint=cc151c7b01dbd27e3efe1866f07fa4d6d65bb6057ef9c117542e206a91e4eda7
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.19.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_19_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_19_decode_layer_call.arg9.boundary_pack.reshard227'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_19_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0201 // fingerprint=81a905b969b0baafe212363d243564a2d74089dda42dbded3e0e3c7cd48649d2
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.19.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard229'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_19_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0202 // fingerprint=d73b7e92aaee82ff572d730f854f64154cb9e3c50b64746f56cef02dc873eed2
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.19.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.19.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_19_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_19_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_19_decode_layer_call.gu_weight_rowmajor.reshard228'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_19_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0203 // fingerprint=672c1d95af6549545b7803cfcdc1a8a1e2c1ec2cf157bcd9a4db0f86f503aeb6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.20.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.input_layernorm.weight')): // bf16<4>[512] name='layer_20_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_20_decode_layer_call.arg2.boundary_pack.reshard230'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_20_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0204 // fingerprint=ee91d8598fae0a6c0dc4baf95030a126391236ca12b8a7cec159a8ad588637e1
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.20.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard231'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_20_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0205 // fingerprint=0e5f557d8a9c96432ceeeafface2cd046cf8b9ca38712201a0571b5124fd2d33
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.20.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard232'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_20_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0206 // fingerprint=40f02850d14b314242e12364137670cfab9c2e49488fa0631a89d31d46c9a1de
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.20.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard233'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_20_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0207 // fingerprint=cc97279c91c2694a996e3ebdb35cb2d727a8edd0cd24acf33e9a5b86f5d985dd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.20.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_20_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_20_decode_layer_call.arg6.boundary_pack.reshard234'
      yield (%1)
    }
    constant_recipe constant_recipe_0208 // fingerprint=b723d4700b27d88e5269f39190a1cd6c3de6acd288ea693bea2d54dca86648f9
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.20.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_20_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_20_decode_layer_call.arg7.boundary_pack.reshard235'
      yield (%1)
    }
    constant_recipe constant_recipe_0209 // fingerprint=05258b615d0f717a128e21c7e4b9f126a7b9b8517f009c47fdda3c040328b2e1
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.20.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard236'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_20_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0210 // fingerprint=3aa8ff8fea4b968ff0f5efb84df5d47b123aa584097b8f0ffd65919913f737ef
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.20.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_20_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_20_decode_layer_call.arg9.boundary_pack.reshard237'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_20_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0211 // fingerprint=4cecea781dd792c7312c8f39d7677e1be5a7ff5e10f367ec4e75f6f6fe32d493
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.20.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard239'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_20_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0212 // fingerprint=086514cd71705b859ef2de9e42855c14ad672eb8af815e64b30174304a902087
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.20.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.20.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_20_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_20_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_20_decode_layer_call.gu_weight_rowmajor.reshard238'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_20_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0213 // fingerprint=5b1e8e4b8550d0e31b82f1894d4ee599fccb52fac2e38d380cdf92e06487aa6c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.21.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.input_layernorm.weight')): // bf16<4>[512] name='layer_21_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_21_decode_layer_call.arg2.boundary_pack.reshard240'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_21_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0214 // fingerprint=924e72e913c578b64afc45f55341db64a31bdddf38aa4c354009a6a11c06bd04
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.21.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard241'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_21_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0215 // fingerprint=43e081f6a36e24e360a8564266d98d07f7378b55902d170e650bb398bb8d9b2c
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.21.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard242'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_21_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0216 // fingerprint=8b7c0fdad6bd20f1d7099fab8f5470f9a8ebeaaa201808674041e0f69b3ba33a
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.21.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard243'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_21_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0217 // fingerprint=846645d7789dfb1c8b870662e4c7a86a2e84efc9872851b6db0f73f5e778124b
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.21.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_21_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_21_decode_layer_call.arg6.boundary_pack.reshard244'
      yield (%1)
    }
    constant_recipe constant_recipe_0218 // fingerprint=59523ee29adfa19219a9b09d16feab43b4afbc75b0d433e06d5289cf780134d4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.21.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_21_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_21_decode_layer_call.arg7.boundary_pack.reshard245'
      yield (%1)
    }
    constant_recipe constant_recipe_0219 // fingerprint=9f2be527225820fc86716de97393168c43ae24060c674cbde4374615f4508565
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.21.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard246'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_21_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0220 // fingerprint=cb7af72d54ef38d4d9dc3051e223998b37bf2f55fdfc897c9bbfddf343212e03
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.21.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_21_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_21_decode_layer_call.arg9.boundary_pack.reshard247'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_21_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0221 // fingerprint=8bc72c8fe8d6e584a25fe64c3ddbe09375979a81814e4fb2488912b0a1dd4569
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.21.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard249'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_21_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0222 // fingerprint=6031976abf43b06d414ea81b6fd0757d8e51f92aa06958b8e21f4558d299039f
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.21.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.21.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_21_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_21_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_21_decode_layer_call.gu_weight_rowmajor.reshard248'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_21_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0223 // fingerprint=e6850103519dbdfa901e4b3123811fbb11590d9bea7a61801785d0e4298b4c19
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.22.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.input_layernorm.weight')): // bf16<4>[512] name='layer_22_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_22_decode_layer_call.arg2.boundary_pack.reshard250'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_22_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0224 // fingerprint=4e53caa277a9a70759a64a4437af9b48495c46ec7f89dff4515a136ff7476691
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.22.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard251'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_22_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0225 // fingerprint=214aed5db7125b965d2016bd100369cff49196580329ddf125c8e138bc8b301a
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.22.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard252'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_22_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0226 // fingerprint=946065c59532181b8bfd532e7685a7808e82248480f14f1734e83c965a144d59
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.22.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard253'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_22_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0227 // fingerprint=7f8bd1facf8b68a6706c6917bcfcd31b3a5985d76cae404564b5ac017c722c14
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.22.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_22_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_22_decode_layer_call.arg6.boundary_pack.reshard254'
      yield (%1)
    }
    constant_recipe constant_recipe_0228 // fingerprint=131c24a4ca3847c33948c505d8aa2c3b7680d073e4e08c27c75a832bbff3a38d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.22.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_22_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_22_decode_layer_call.arg7.boundary_pack.reshard255'
      yield (%1)
    }
    constant_recipe constant_recipe_0229 // fingerprint=5119960b20aefcb89b910ccfb6219a157388a827b27a4a5a6595e3d70c5f1576
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.22.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard256'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_22_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0230 // fingerprint=aba514c0f8bc9fb5d6a3f7aa357ec4d9c76a9e47535a793d61a703fcdefc1fe1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.22.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_22_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_22_decode_layer_call.arg9.boundary_pack.reshard257'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_22_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0231 // fingerprint=6d590c08d8afacc59ec92ae7dc98f66004548eeda7035e42901df7ffa35c5bc4
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.22.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard259'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_22_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0232 // fingerprint=34e13235cbbd852049a11a0675ad031a5037b52df4d32e6fc64f41c868b4d48e
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.22.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.22.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_22_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_22_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_22_decode_layer_call.gu_weight_rowmajor.reshard258'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_22_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0233 // fingerprint=a5f81ae71dda72e151b72cf9197e885b3b8ccc28b5774634ed3bfe61583005d4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.23.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.input_layernorm.weight')): // bf16<4>[512] name='layer_23_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_23_decode_layer_call.arg2.boundary_pack.reshard260'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_23_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0234 // fingerprint=25f929f80064bb4dcc9fa71c287a520ba66281fae17a503b119f67532c01527d
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.23.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard261'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_23_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0235 // fingerprint=7c5860ecafa3d2e79ea97c4dd2e137673b023c34ff3ab9e88704ccceae54d8d4
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.23.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard262'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_23_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0236 // fingerprint=8fa368ec1bfc368d6b9dc7423e95495f0589e61860ba5b4a1078783f6e4a8bd4
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.23.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard263'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_23_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0237 // fingerprint=c6b1925b7b4a07fec1c025d590f5e9b74b1e7386730716d75a4518eeb075a302
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.23.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_23_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_23_decode_layer_call.arg6.boundary_pack.reshard264'
      yield (%1)
    }
    constant_recipe constant_recipe_0238 // fingerprint=36a3d38c27943cec1e50edf0e26112e8e2ebc7312149ee8c5916dd823556021c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.23.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_23_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_23_decode_layer_call.arg7.boundary_pack.reshard265'
      yield (%1)
    }
    constant_recipe constant_recipe_0239 // fingerprint=f513fa38e7b913535372b4216f51f7a6be8cef1f1c3ab059f0d53907e5716728
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.23.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard266'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_23_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0240 // fingerprint=79d714e2fdd6780ed537c83b73238f6ad3ed808046e49c89c118227386acdd5d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.23.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_23_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_23_decode_layer_call.arg9.boundary_pack.reshard267'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_23_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0241 // fingerprint=0cdf24d4f6fa37e07e89e0e224e2e92f095f641383c81cd4a2c64c3357bcab73
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.23.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard269'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_23_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0242 // fingerprint=c2e1973b2632d928cc6b674bfd452452bc375f723e39c8028eccdd1fc3a099cf
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.23.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.23.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_23_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_23_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_23_decode_layer_call.gu_weight_rowmajor.reshard268'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_23_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0243 // fingerprint=d9c899812e20c7772fdb8bda1d91e21a60b2426014b62c6bd10cbd2cfd504834
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.24.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.input_layernorm.weight')): // bf16<4>[512] name='layer_24_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_24_decode_layer_call.arg2.boundary_pack.reshard270'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_24_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0244 // fingerprint=f477034cf0f48dfceacd0937b4423c6cbdbf6785d135dcc8d903267715de115c
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.24.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard271'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_24_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0245 // fingerprint=76ce4928e34da325eb3ceb3f0395e66148e7d67680e8dd74c1d4706def05cc89
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.24.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard272'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_24_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0246 // fingerprint=ffc22786b7e1dd35c14ae8b2faea2697ff644f8a01ff03f62987b0ec6004010c
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.24.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard273'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_24_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0247 // fingerprint=01741f03d44130c066b9b26437656127bc2671881a1f68508893d3c309fc060c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.24.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_24_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_24_decode_layer_call.arg6.boundary_pack.reshard274'
      yield (%1)
    }
    constant_recipe constant_recipe_0248 // fingerprint=cb7eea23432de78782b9cb7cb0d9a56ca8c9e12fc6f97eebaa7b1754c556bce9
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.24.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_24_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_24_decode_layer_call.arg7.boundary_pack.reshard275'
      yield (%1)
    }
    constant_recipe constant_recipe_0249 // fingerprint=55ee05820986f656e828231bcc425e72b130e6defb55512f237e57aabb8db675
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.24.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard276'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_24_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0250 // fingerprint=15d9a6a05d86f3fcb11b6cb3b081972fa55a112c09c3e1efc65bbfecaa7ebfe0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.24.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_24_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_24_decode_layer_call.arg9.boundary_pack.reshard277'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_24_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0251 // fingerprint=2a5462339f30b8de02ec01f1d641358015f16175e6e6965f9d6718e71c2937d4
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.24.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard279'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_24_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0252 // fingerprint=ec45126eca7e921aedaafe78d73d62c18888395befc0d2232597f6613291b638
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.24.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.24.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_24_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_24_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_24_decode_layer_call.gu_weight_rowmajor.reshard278'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_24_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0253 // fingerprint=a146661de56ef9f23c217b05634f7c3b0fe9458481bb1596cc5d9ae8b46bd44e
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.25.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.input_layernorm.weight')): // bf16<4>[512] name='layer_25_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_25_decode_layer_call.arg2.boundary_pack.reshard280'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_25_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0254 // fingerprint=4ca5b61444cbbcee09b125a09254de61e20156c681c925176431274509e80741
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.25.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard281'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_25_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0255 // fingerprint=c83bc179fcbc9ad667ee3e61a124ae4331e7aebc1d867e9aef9035049c37cf9b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.25.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard282'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_25_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0256 // fingerprint=adc92bcaa99567b2288567aa4a241f4c627856872193de4c7ab52d49b56025dc
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.25.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard283'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_25_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0257 // fingerprint=72544951d76a40a4d5ab5180eb34565cdeace89817e7e6a457cb06af33cd3359
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.25.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_25_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_25_decode_layer_call.arg6.boundary_pack.reshard284'
      yield (%1)
    }
    constant_recipe constant_recipe_0258 // fingerprint=c4808b1a2bb0a01f77c64f2438dce72115ad3091f41ce876286536e756db728c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.25.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_25_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_25_decode_layer_call.arg7.boundary_pack.reshard285'
      yield (%1)
    }
    constant_recipe constant_recipe_0259 // fingerprint=4953a59d14ddbaada8dc413b94a9206df981e93ee4692752ef23308d29479ab9
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.25.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard286'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_25_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0260 // fingerprint=e94476c16a559660c91130a9572d9c2a10af417930cae5afd431d5164ff190cd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.25.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_25_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_25_decode_layer_call.arg9.boundary_pack.reshard287'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_25_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0261 // fingerprint=ea5e82d498b7ff69890c9bc83995731a9abd7db654cc8a9686b54a360288f0a0
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.25.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard289'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_25_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0262 // fingerprint=281fe549901d005a6a04db3f5bd5ef7f42d7f7c7bf8542a3a65da859df19a096
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.25.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.25.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_25_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_25_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_25_decode_layer_call.gu_weight_rowmajor.reshard288'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_25_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0263 // fingerprint=25e1b00911b7436ff77bd2248d6b5a0cc060924c3eb9e6f904d23844656db209
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.26.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.input_layernorm.weight')): // bf16<4>[512] name='layer_26_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_26_decode_layer_call.arg2.boundary_pack.reshard290'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_26_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0264 // fingerprint=04dd4e3d4c145a9773f5d0ec2cf0520f3833679a893f0c0167e21ad7a8db5130
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.26.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard291'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_26_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0265 // fingerprint=c128c92179b310cadd7ae99a8900b3b0c5fa5cd50676d98a182b690a95dcec73
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.26.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard292'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_26_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0266 // fingerprint=5496c650759f613baf6378ef11f42fbbed503405986fd203c3e62c3a4dfe83df
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.26.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard293'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_26_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0267 // fingerprint=a654719eeb0879ef06d4b2afc33dc2f08f6d40e06b256804a799e1dd3ace85a0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.26.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_26_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_26_decode_layer_call.arg6.boundary_pack.reshard294'
      yield (%1)
    }
    constant_recipe constant_recipe_0268 // fingerprint=5258f7b9ac45bc576b6aa7971cb265a915aff95b5f619bc4a73a4b5be173b1cb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.26.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_26_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_26_decode_layer_call.arg7.boundary_pack.reshard295'
      yield (%1)
    }
    constant_recipe constant_recipe_0269 // fingerprint=a4415fa2e016515e7f5a1b662c5b48c4277a1ab0fe7436739408f64c8d9bd3a1
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.26.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard296'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_26_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0270 // fingerprint=efc11e28cc4112f4d3828129766b52eb4bc103d8b22ebc8fef33f7b8af1fe0dd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.26.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_26_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_26_decode_layer_call.arg9.boundary_pack.reshard297'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_26_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0271 // fingerprint=01485f65315a0a58125f7b1601ffc18960366355cbf36ef0a58225055cca1fdd
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.26.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard299'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_26_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0272 // fingerprint=fb0e3566373cd132e042b723f4f5bf8eca38df88facb1648250fc81a884e8f22
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.26.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.26.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_26_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_26_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_26_decode_layer_call.gu_weight_rowmajor.reshard298'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_26_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0273 // fingerprint=31df1e26a0c7a6a3f94760b3f3dac5d2602a5dbdcad6b958ef71e4e63bc6cac1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.27.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.input_layernorm.weight')): // bf16<4>[512] name='layer_27_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_27_decode_layer_call.arg2.boundary_pack.reshard300'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(64)@[0])), [64@y], } name='layer_27_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0274 // fingerprint=82898408b84e35c92c9bab75142eebebce4b9e0f22a6a9cc190b84b1bd557e8e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.27.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard301'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], } name='layer_27_decode_layer_call.arg3.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0275 // fingerprint=ffcd4e69fdc125511f60bcf1c1b22f2f7bc63609fd95367df66ce0f3085d5bad
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.27.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard302'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_27_decode_layer_call.arg4.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0276 // fingerprint=bae829d7e546423bba4b365e6eb1fa9b9f8f7f2155cc75da3e475436a0b79444
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[1024,2048], 'layers.27.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[1024,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %5 = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %6 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: , %5): // {bf16<8,2,8>[128,128], (B,B), [128,128], } name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard303'
      %7 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: , %6): // {bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], } name='layer_27_decode_layer_call.arg5.distributed_boundary_reshard'
      yield (%7)
    }
    constant_recipe constant_recipe_0277 // fingerprint=f4e9f3abf74349cd569d121beb7ce7de2ed483a79017d45bb5d777da7af43ab1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.27.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_27_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_27_decode_layer_call.arg6.boundary_pack.reshard304'
      yield (%1)
    }
    constant_recipe constant_recipe_0278 // fingerprint=dd7939992141421cd05c28b40b2652f568f9421ffdb202a860add4de855d7e0b
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], WeightRef(bf16[128], 'layers.27.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_27_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<8>[16], (B), [y:8,x:16], Partial: , %0): // {bf16<8>[16], (B), [16], } name='layer_27_decode_layer_call.arg7.boundary_pack.reshard305'
      yield (%1)
    }
    constant_recipe constant_recipe_0279 // fingerprint=7ebcef0466c90e6efd563c02484283dcc20b28c34c47a4a3bf3b0ed4e608f860
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,2048], 'layers.27.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[128,256], (B,B), [128,256], } name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard306'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], } name='layer_27_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0280 // fingerprint=a99e58b02805ec3378b5534be6d06e0f0a2716625ac433d8ecb31b7db4237ffb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'layers.27.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_27_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (B), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (B), [512], } name='layer_27_decode_layer_call.arg9.boundary_pack.reshard307'
      %2 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %1): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='layer_27_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    constant_recipe constant_recipe_0281 // fingerprint=374182c3d80bd557f9f09c2bcdf1291b50994827d0b641784f465a56be31e9f7
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], WeightRef(bf16[2048,6144], 'layers.27.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: , %3): // {bf16<8,2,8>[384,256], (B,B), [384,256], } name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard309'
      %5 = Distributed.ShardedView(NewType: bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: , %4): // {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], } name='layer_27_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    constant_recipe constant_recipe_0282 // fingerprint=18570d7a91daf5745be5a4db5e66c96926c77003e7d1938ce4f8a07359a4cf4c
    {
      %0 = Tensors.Concat(Axis: 0, WeightRef(bf16[6144,2048], 'layers.27.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.mlp.gate_proj.weight'), WeightRef(bf16[6144,2048], 'layers.27.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_27_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_27_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: bf16[2048,12288], (B,B), [y:8,x:16], Partial: , %1): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_27_decode_layer_call.gu_weight_rowmajor.reshard308'
      %3 = Distributed.Boxing(NewType: bf16[2048,12288], (B,B), [y:48,x:2], Partial: , %2): // {bf16[2048,12288], (B,B), [2048,12288], } name='layer_27_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    constant_recipe constant_recipe_0283 // fingerprint=aa591a81a5de287d0a3c96d795aa00689da07bf8bac76c682dbfa8ebbbc3b77c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], WeightRef(bf16[2048], 'norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.norm.weight')): // bf16<4>[512] name='final_norm.vectorized.pack2'
      %1 = Distributed.ShardedView(NewType: bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: , %0): // {bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], } name='final_norm.vectorized.pack2.reshard311'
      yield (%1)
    }
    %w0 = None(): // none name='decode_layer_qkv_none'
  }
  // compute
  %0   = Distributed.ShardedView(NewType: i32[1], (B), [y:8,x:16], Partial: , %input_ids): // {i32[1], (B), [1], } name='input_ids.reshard15'
  %1   = NN.Embedding(PaddingIdx: None, PostOps: [fusion(value: bf16<4>[1,512]) => NTT.VectorizedCast(value, NewType: {elem_type: 'float32', kind: 'vector', lanes: [4]}, RoundingContract: 'preserve', VectorizeAxes: [1])], %0, ConstAssetRef({bf16<4>[151936,512], (B,S(C(4)@[0,1])), [151936,4@y@x], }, Recipe: 'constant_recipe_0000', Output: 'token_embedding.packed_weight.reshard16')): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='token_embedding.packed'
  %2   = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %1): // {f32<4>[1,512], (B,B), [1,512], } name='token_embedding.packed.reshard17'
  %3   = NN.RotaryEmbedding(AttentionScaling: 1.0, HeadDim: 128, OutputDtype: 'bfloat16', OutputLanes: [2, 8], Theta: 1000000.0, %2, %paged_attention_kv_cache): // ({bf16<2,8>[1,1,8], (B,B,B), [1,1,8], }, {bf16<2,8>[1,1,8], (B,B,B), [1,1,8], }) !read<paged_attention_kv_cache> name='main_rotary_embedding.packed'
  %4   = Tensors.Bitcast(Dtype: {elem_type: 'float32', kind: 'vector', lanes: [4]}, %2): // {f32<4>[1,512], (B,B), [1,512], } name='layer_0_decode_layer_call.arg0.boundary_pack'
  %5   = GetItem(%3, 0): // {bf16<2,8>[1,1,8], (B,B,B), [1,1,8], } name='layer_0_decode_layer_call.arg15.boundary_pack'
  %6   = GetItem(%3, 1): // {bf16<2,8>[1,1,8], (B,B,B), [1,1,8], } name='layer_0_decode_layer_call.arg16.boundary_pack'
  %7   = NN.NormStats(Axis: 1, UseMean: False, %4): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_0_decode_layer_call.norm_stats'
  %8   = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %4): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_0_decode_layer_call.arg0.distributed_boundary_reshard'
  %9   = Call(Callee: 'decode_layer', %8, %paged_attention_kv_cache, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0003', Output: 'layer_0_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0004', Output: 'layer_0_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0005', Output: 'layer_0_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0006', Output: 'layer_0_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0007', Output: 'layer_0_decode_layer_call.arg6.boundary_pack.reshard34'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0008', Output: 'layer_0_decode_layer_call.arg7.boundary_pack.reshard35'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0009', Output: 'layer_0_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0010', Output: 'layer_0_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0012', Output: 'layer_0_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0011', Output: 'layer_0_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 0), const(bool[] : False), %5, %6, %7): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_0_decode_layer_call'
  %10  = GetItem(%9, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_0_decoder_output.boundary_packed.distributed_boundary_abi'
  %11  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %10): // {f32<4>[1,512], (B,B), [1,512], } name='layer_0_decoder_output.boundary_packed'
  %12  = GetItem(%9, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_0_updated_state'
  %13  = GetItem(%9, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_1_decode_layer_call.norm_stats'
  %14  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %11): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_1_decode_layer_call.arg0.distributed_boundary_reshard'
  %15  = Call(Callee: 'decode_layer', %14, %12, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0013', Output: 'layer_1_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0014', Output: 'layer_1_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0015', Output: 'layer_1_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0016', Output: 'layer_1_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0017', Output: 'layer_1_decode_layer_call.arg6.boundary_pack.reshard44'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0018', Output: 'layer_1_decode_layer_call.arg7.boundary_pack.reshard45'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0019', Output: 'layer_1_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0020', Output: 'layer_1_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0022', Output: 'layer_1_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0021', Output: 'layer_1_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 1), const(bool[] : False), %5, %6, %13): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_1_decode_layer_call'
  %16  = GetItem(%15, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_1_decoder_output.boundary_packed.distributed_boundary_abi'
  %17  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %16): // {f32<4>[1,512], (B,B), [1,512], } name='layer_1_decoder_output.boundary_packed'
  %18  = GetItem(%15, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_1_updated_state'
  %19  = GetItem(%15, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_2_decode_layer_call.norm_stats'
  %20  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %17): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_2_decode_layer_call.arg0.distributed_boundary_reshard'
  %21  = Call(Callee: 'decode_layer', %20, %18, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0023', Output: 'layer_2_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0024', Output: 'layer_2_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0025', Output: 'layer_2_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0026', Output: 'layer_2_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0027', Output: 'layer_2_decode_layer_call.arg6.boundary_pack.reshard54'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0028', Output: 'layer_2_decode_layer_call.arg7.boundary_pack.reshard55'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0029', Output: 'layer_2_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0030', Output: 'layer_2_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0032', Output: 'layer_2_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0031', Output: 'layer_2_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 2), const(bool[] : False), %5, %6, %19): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_2_decode_layer_call'
  %22  = GetItem(%21, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_2_decoder_output.boundary_packed.distributed_boundary_abi'
  %23  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %22): // {f32<4>[1,512], (B,B), [1,512], } name='layer_2_decoder_output.boundary_packed'
  %24  = GetItem(%21, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_2_updated_state'
  %25  = GetItem(%21, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_3_decode_layer_call.norm_stats'
  %26  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %23): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_3_decode_layer_call.arg0.distributed_boundary_reshard'
  %27  = Call(Callee: 'decode_layer', %26, %24, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0033', Output: 'layer_3_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0034', Output: 'layer_3_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0035', Output: 'layer_3_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0036', Output: 'layer_3_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0037', Output: 'layer_3_decode_layer_call.arg6.boundary_pack.reshard64'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0038', Output: 'layer_3_decode_layer_call.arg7.boundary_pack.reshard65'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0039', Output: 'layer_3_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0040', Output: 'layer_3_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0042', Output: 'layer_3_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0041', Output: 'layer_3_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 3), const(bool[] : False), %5, %6, %25): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_3_decode_layer_call'
  %28  = GetItem(%27, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_3_decoder_output.boundary_packed.distributed_boundary_abi'
  %29  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %28): // {f32<4>[1,512], (B,B), [1,512], } name='layer_3_decoder_output.boundary_packed'
  %30  = GetItem(%27, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_3_updated_state'
  %31  = GetItem(%27, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_4_decode_layer_call.norm_stats'
  %32  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %29): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_4_decode_layer_call.arg0.distributed_boundary_reshard'
  %33  = Call(Callee: 'decode_layer', %32, %30, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0043', Output: 'layer_4_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0044', Output: 'layer_4_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0045', Output: 'layer_4_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0046', Output: 'layer_4_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0047', Output: 'layer_4_decode_layer_call.arg6.boundary_pack.reshard74'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0048', Output: 'layer_4_decode_layer_call.arg7.boundary_pack.reshard75'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0049', Output: 'layer_4_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0050', Output: 'layer_4_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0052', Output: 'layer_4_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0051', Output: 'layer_4_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 4), const(bool[] : False), %5, %6, %31): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_4_decode_layer_call'
  %34  = GetItem(%33, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_4_decoder_output.boundary_packed.distributed_boundary_abi'
  %35  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %34): // {f32<4>[1,512], (B,B), [1,512], } name='layer_4_decoder_output.boundary_packed'
  %36  = GetItem(%33, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_4_updated_state'
  %37  = GetItem(%33, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_5_decode_layer_call.norm_stats'
  %38  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %35): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_5_decode_layer_call.arg0.distributed_boundary_reshard'
  %39  = Call(Callee: 'decode_layer', %38, %36, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0053', Output: 'layer_5_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0054', Output: 'layer_5_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0055', Output: 'layer_5_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0056', Output: 'layer_5_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0057', Output: 'layer_5_decode_layer_call.arg6.boundary_pack.reshard84'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0058', Output: 'layer_5_decode_layer_call.arg7.boundary_pack.reshard85'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0059', Output: 'layer_5_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0060', Output: 'layer_5_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0062', Output: 'layer_5_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0061', Output: 'layer_5_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 5), const(bool[] : False), %5, %6, %37): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_5_decode_layer_call'
  %40  = GetItem(%39, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_5_decoder_output.boundary_packed.distributed_boundary_abi'
  %41  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %40): // {f32<4>[1,512], (B,B), [1,512], } name='layer_5_decoder_output.boundary_packed'
  %42  = GetItem(%39, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_5_updated_state'
  %43  = GetItem(%39, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_6_decode_layer_call.norm_stats'
  %44  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %41): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_6_decode_layer_call.arg0.distributed_boundary_reshard'
  %45  = Call(Callee: 'decode_layer', %44, %42, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0063', Output: 'layer_6_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0064', Output: 'layer_6_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0065', Output: 'layer_6_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0066', Output: 'layer_6_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0067', Output: 'layer_6_decode_layer_call.arg6.boundary_pack.reshard94'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0068', Output: 'layer_6_decode_layer_call.arg7.boundary_pack.reshard95'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0069', Output: 'layer_6_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0070', Output: 'layer_6_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0072', Output: 'layer_6_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0071', Output: 'layer_6_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 6), const(bool[] : False), %5, %6, %43): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_6_decode_layer_call'
  %46  = GetItem(%45, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_6_decoder_output.boundary_packed.distributed_boundary_abi'
  %47  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %46): // {f32<4>[1,512], (B,B), [1,512], } name='layer_6_decoder_output.boundary_packed'
  %48  = GetItem(%45, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_6_updated_state'
  %49  = GetItem(%45, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_7_decode_layer_call.norm_stats'
  %50  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %47): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_7_decode_layer_call.arg0.distributed_boundary_reshard'
  %51  = Call(Callee: 'decode_layer', %50, %48, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0073', Output: 'layer_7_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0074', Output: 'layer_7_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0075', Output: 'layer_7_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0076', Output: 'layer_7_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0077', Output: 'layer_7_decode_layer_call.arg6.boundary_pack.reshard104'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0078', Output: 'layer_7_decode_layer_call.arg7.boundary_pack.reshard105'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0079', Output: 'layer_7_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0080', Output: 'layer_7_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0082', Output: 'layer_7_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0081', Output: 'layer_7_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 7), const(bool[] : False), %5, %6, %49): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_7_decode_layer_call'
  %52  = GetItem(%51, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_7_decoder_output.boundary_packed.distributed_boundary_abi'
  %53  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %52): // {f32<4>[1,512], (B,B), [1,512], } name='layer_7_decoder_output.boundary_packed'
  %54  = GetItem(%51, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_7_updated_state'
  %55  = GetItem(%51, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_8_decode_layer_call.norm_stats'
  %56  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %53): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_8_decode_layer_call.arg0.distributed_boundary_reshard'
  %57  = Call(Callee: 'decode_layer', %56, %54, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0083', Output: 'layer_8_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0084', Output: 'layer_8_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0085', Output: 'layer_8_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0086', Output: 'layer_8_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0087', Output: 'layer_8_decode_layer_call.arg6.boundary_pack.reshard114'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0088', Output: 'layer_8_decode_layer_call.arg7.boundary_pack.reshard115'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0089', Output: 'layer_8_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0090', Output: 'layer_8_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0092', Output: 'layer_8_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0091', Output: 'layer_8_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 8), const(bool[] : False), %5, %6, %55): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_8_decode_layer_call'
  %58  = GetItem(%57, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_8_decoder_output.boundary_packed.distributed_boundary_abi'
  %59  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %58): // {f32<4>[1,512], (B,B), [1,512], } name='layer_8_decoder_output.boundary_packed'
  %60  = GetItem(%57, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_8_updated_state'
  %61  = GetItem(%57, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_9_decode_layer_call.norm_stats'
  %62  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %59): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_9_decode_layer_call.arg0.distributed_boundary_reshard'
  %63  = Call(Callee: 'decode_layer', %62, %60, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0093', Output: 'layer_9_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0094', Output: 'layer_9_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0095', Output: 'layer_9_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0096', Output: 'layer_9_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0097', Output: 'layer_9_decode_layer_call.arg6.boundary_pack.reshard124'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0098', Output: 'layer_9_decode_layer_call.arg7.boundary_pack.reshard125'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0099', Output: 'layer_9_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0100', Output: 'layer_9_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0102', Output: 'layer_9_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0101', Output: 'layer_9_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 9), const(bool[] : False), %5, %6, %61): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_9_decode_layer_call'
  %64  = GetItem(%63, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_9_decoder_output.boundary_packed.distributed_boundary_abi'
  %65  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %64): // {f32<4>[1,512], (B,B), [1,512], } name='layer_9_decoder_output.boundary_packed'
  %66  = GetItem(%63, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_9_updated_state'
  %67  = GetItem(%63, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_10_decode_layer_call.norm_stats'
  %68  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %65): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_10_decode_layer_call.arg0.distributed_boundary_reshard'
  %69  = Call(Callee: 'decode_layer', %68, %66, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0103', Output: 'layer_10_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0104', Output: 'layer_10_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0105', Output: 'layer_10_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0106', Output: 'layer_10_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0107', Output: 'layer_10_decode_layer_call.arg6.boundary_pack.reshard134'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0108', Output: 'layer_10_decode_layer_call.arg7.boundary_pack.reshard135'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0109', Output: 'layer_10_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0110', Output: 'layer_10_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0112', Output: 'layer_10_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0111', Output: 'layer_10_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 10), const(bool[] : False), %5, %6, %67): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_10_decode_layer_call'
  %70  = GetItem(%69, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_10_decoder_output.boundary_packed.distributed_boundary_abi'
  %71  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %70): // {f32<4>[1,512], (B,B), [1,512], } name='layer_10_decoder_output.boundary_packed'
  %72  = GetItem(%69, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_10_updated_state'
  %73  = GetItem(%69, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_11_decode_layer_call.norm_stats'
  %74  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %71): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_11_decode_layer_call.arg0.distributed_boundary_reshard'
  %75  = Call(Callee: 'decode_layer', %74, %72, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0113', Output: 'layer_11_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0114', Output: 'layer_11_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0115', Output: 'layer_11_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0116', Output: 'layer_11_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0117', Output: 'layer_11_decode_layer_call.arg6.boundary_pack.reshard144'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0118', Output: 'layer_11_decode_layer_call.arg7.boundary_pack.reshard145'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0119', Output: 'layer_11_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0120', Output: 'layer_11_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0122', Output: 'layer_11_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0121', Output: 'layer_11_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 11), const(bool[] : False), %5, %6, %73): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_11_decode_layer_call'
  %76  = GetItem(%75, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_11_decoder_output.boundary_packed.distributed_boundary_abi'
  %77  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %76): // {f32<4>[1,512], (B,B), [1,512], } name='layer_11_decoder_output.boundary_packed'
  %78  = GetItem(%75, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_11_updated_state'
  %79  = GetItem(%75, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_12_decode_layer_call.norm_stats'
  %80  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %77): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_12_decode_layer_call.arg0.distributed_boundary_reshard'
  %81  = Call(Callee: 'decode_layer', %80, %78, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0123', Output: 'layer_12_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0124', Output: 'layer_12_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0125', Output: 'layer_12_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0126', Output: 'layer_12_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0127', Output: 'layer_12_decode_layer_call.arg6.boundary_pack.reshard154'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0128', Output: 'layer_12_decode_layer_call.arg7.boundary_pack.reshard155'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0129', Output: 'layer_12_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0130', Output: 'layer_12_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0132', Output: 'layer_12_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0131', Output: 'layer_12_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 12), const(bool[] : False), %5, %6, %79): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_12_decode_layer_call'
  %82  = GetItem(%81, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_12_decoder_output.boundary_packed.distributed_boundary_abi'
  %83  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %82): // {f32<4>[1,512], (B,B), [1,512], } name='layer_12_decoder_output.boundary_packed'
  %84  = GetItem(%81, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_12_updated_state'
  %85  = GetItem(%81, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_13_decode_layer_call.norm_stats'
  %86  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %83): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_13_decode_layer_call.arg0.distributed_boundary_reshard'
  %87  = Call(Callee: 'decode_layer', %86, %84, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0133', Output: 'layer_13_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0134', Output: 'layer_13_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0135', Output: 'layer_13_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0136', Output: 'layer_13_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0137', Output: 'layer_13_decode_layer_call.arg6.boundary_pack.reshard164'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0138', Output: 'layer_13_decode_layer_call.arg7.boundary_pack.reshard165'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0139', Output: 'layer_13_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0140', Output: 'layer_13_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0142', Output: 'layer_13_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0141', Output: 'layer_13_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 13), const(bool[] : False), %5, %6, %85): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_13_decode_layer_call'
  %88  = GetItem(%87, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_13_decoder_output.boundary_packed.distributed_boundary_abi'
  %89  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %88): // {f32<4>[1,512], (B,B), [1,512], } name='layer_13_decoder_output.boundary_packed'
  %90  = GetItem(%87, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_13_updated_state'
  %91  = GetItem(%87, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_14_decode_layer_call.norm_stats'
  %92  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %89): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_14_decode_layer_call.arg0.distributed_boundary_reshard'
  %93  = Call(Callee: 'decode_layer', %92, %90, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0143', Output: 'layer_14_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0144', Output: 'layer_14_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0145', Output: 'layer_14_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0146', Output: 'layer_14_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0147', Output: 'layer_14_decode_layer_call.arg6.boundary_pack.reshard174'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0148', Output: 'layer_14_decode_layer_call.arg7.boundary_pack.reshard175'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0149', Output: 'layer_14_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0150', Output: 'layer_14_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0152', Output: 'layer_14_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0151', Output: 'layer_14_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 14), const(bool[] : False), %5, %6, %91): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_14_decode_layer_call'
  %94  = GetItem(%93, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_14_decoder_output.boundary_packed.distributed_boundary_abi'
  %95  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %94): // {f32<4>[1,512], (B,B), [1,512], } name='layer_14_decoder_output.boundary_packed'
  %96  = GetItem(%93, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_14_updated_state'
  %97  = GetItem(%93, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_15_decode_layer_call.norm_stats'
  %98  = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %95): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_15_decode_layer_call.arg0.distributed_boundary_reshard'
  %99  = Call(Callee: 'decode_layer', %98, %96, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0153', Output: 'layer_15_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0154', Output: 'layer_15_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0155', Output: 'layer_15_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0156', Output: 'layer_15_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0157', Output: 'layer_15_decode_layer_call.arg6.boundary_pack.reshard184'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0158', Output: 'layer_15_decode_layer_call.arg7.boundary_pack.reshard185'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0159', Output: 'layer_15_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0160', Output: 'layer_15_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0162', Output: 'layer_15_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0161', Output: 'layer_15_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 15), const(bool[] : False), %5, %6, %97): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_15_decode_layer_call'
  %100 = GetItem(%99, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_15_decoder_output.boundary_packed.distributed_boundary_abi'
  %101 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %100): // {f32<4>[1,512], (B,B), [1,512], } name='layer_15_decoder_output.boundary_packed'
  %102 = GetItem(%99, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_15_updated_state'
  %103 = GetItem(%99, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_16_decode_layer_call.norm_stats'
  %104 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %101): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_16_decode_layer_call.arg0.distributed_boundary_reshard'
  %105 = Call(Callee: 'decode_layer', %104, %102, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0163', Output: 'layer_16_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0164', Output: 'layer_16_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0165', Output: 'layer_16_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0166', Output: 'layer_16_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0167', Output: 'layer_16_decode_layer_call.arg6.boundary_pack.reshard194'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0168', Output: 'layer_16_decode_layer_call.arg7.boundary_pack.reshard195'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0169', Output: 'layer_16_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0170', Output: 'layer_16_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0172', Output: 'layer_16_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0171', Output: 'layer_16_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 16), const(bool[] : False), %5, %6, %103): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_16_decode_layer_call'
  %106 = GetItem(%105, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_16_decoder_output.boundary_packed.distributed_boundary_abi'
  %107 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %106): // {f32<4>[1,512], (B,B), [1,512], } name='layer_16_decoder_output.boundary_packed'
  %108 = GetItem(%105, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_16_updated_state'
  %109 = GetItem(%105, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_17_decode_layer_call.norm_stats'
  %110 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %107): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_17_decode_layer_call.arg0.distributed_boundary_reshard'
  %111 = Call(Callee: 'decode_layer', %110, %108, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0173', Output: 'layer_17_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0174', Output: 'layer_17_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0175', Output: 'layer_17_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0176', Output: 'layer_17_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0177', Output: 'layer_17_decode_layer_call.arg6.boundary_pack.reshard204'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0178', Output: 'layer_17_decode_layer_call.arg7.boundary_pack.reshard205'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0179', Output: 'layer_17_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0180', Output: 'layer_17_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0182', Output: 'layer_17_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0181', Output: 'layer_17_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 17), const(bool[] : False), %5, %6, %109): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_17_decode_layer_call'
  %112 = GetItem(%111, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_17_decoder_output.boundary_packed.distributed_boundary_abi'
  %113 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %112): // {f32<4>[1,512], (B,B), [1,512], } name='layer_17_decoder_output.boundary_packed'
  %114 = GetItem(%111, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_17_updated_state'
  %115 = GetItem(%111, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_18_decode_layer_call.norm_stats'
  %116 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %113): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_18_decode_layer_call.arg0.distributed_boundary_reshard'
  %117 = Call(Callee: 'decode_layer', %116, %114, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0183', Output: 'layer_18_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0184', Output: 'layer_18_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0185', Output: 'layer_18_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0186', Output: 'layer_18_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0187', Output: 'layer_18_decode_layer_call.arg6.boundary_pack.reshard214'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0188', Output: 'layer_18_decode_layer_call.arg7.boundary_pack.reshard215'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0189', Output: 'layer_18_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0190', Output: 'layer_18_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0192', Output: 'layer_18_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0191', Output: 'layer_18_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 18), const(bool[] : False), %5, %6, %115): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_18_decode_layer_call'
  %118 = GetItem(%117, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_18_decoder_output.boundary_packed.distributed_boundary_abi'
  %119 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %118): // {f32<4>[1,512], (B,B), [1,512], } name='layer_18_decoder_output.boundary_packed'
  %120 = GetItem(%117, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_18_updated_state'
  %121 = GetItem(%117, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_19_decode_layer_call.norm_stats'
  %122 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %119): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_19_decode_layer_call.arg0.distributed_boundary_reshard'
  %123 = Call(Callee: 'decode_layer', %122, %120, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0193', Output: 'layer_19_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0194', Output: 'layer_19_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0195', Output: 'layer_19_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0196', Output: 'layer_19_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0197', Output: 'layer_19_decode_layer_call.arg6.boundary_pack.reshard224'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0198', Output: 'layer_19_decode_layer_call.arg7.boundary_pack.reshard225'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0199', Output: 'layer_19_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0200', Output: 'layer_19_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0202', Output: 'layer_19_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0201', Output: 'layer_19_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 19), const(bool[] : False), %5, %6, %121): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_19_decode_layer_call'
  %124 = GetItem(%123, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_19_decoder_output.boundary_packed.distributed_boundary_abi'
  %125 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %124): // {f32<4>[1,512], (B,B), [1,512], } name='layer_19_decoder_output.boundary_packed'
  %126 = GetItem(%123, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_19_updated_state'
  %127 = GetItem(%123, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_20_decode_layer_call.norm_stats'
  %128 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %125): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_20_decode_layer_call.arg0.distributed_boundary_reshard'
  %129 = Call(Callee: 'decode_layer', %128, %126, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0203', Output: 'layer_20_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0204', Output: 'layer_20_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0205', Output: 'layer_20_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0206', Output: 'layer_20_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0207', Output: 'layer_20_decode_layer_call.arg6.boundary_pack.reshard234'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0208', Output: 'layer_20_decode_layer_call.arg7.boundary_pack.reshard235'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0209', Output: 'layer_20_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0210', Output: 'layer_20_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0212', Output: 'layer_20_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0211', Output: 'layer_20_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 20), const(bool[] : False), %5, %6, %127): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_20_decode_layer_call'
  %130 = GetItem(%129, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_20_decoder_output.boundary_packed.distributed_boundary_abi'
  %131 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %130): // {f32<4>[1,512], (B,B), [1,512], } name='layer_20_decoder_output.boundary_packed'
  %132 = GetItem(%129, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_20_updated_state'
  %133 = GetItem(%129, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_21_decode_layer_call.norm_stats'
  %134 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %131): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_21_decode_layer_call.arg0.distributed_boundary_reshard'
  %135 = Call(Callee: 'decode_layer', %134, %132, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0213', Output: 'layer_21_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0214', Output: 'layer_21_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0215', Output: 'layer_21_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0216', Output: 'layer_21_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0217', Output: 'layer_21_decode_layer_call.arg6.boundary_pack.reshard244'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0218', Output: 'layer_21_decode_layer_call.arg7.boundary_pack.reshard245'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0219', Output: 'layer_21_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0220', Output: 'layer_21_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0222', Output: 'layer_21_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0221', Output: 'layer_21_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 21), const(bool[] : False), %5, %6, %133): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_21_decode_layer_call'
  %136 = GetItem(%135, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_21_decoder_output.boundary_packed.distributed_boundary_abi'
  %137 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %136): // {f32<4>[1,512], (B,B), [1,512], } name='layer_21_decoder_output.boundary_packed'
  %138 = GetItem(%135, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_21_updated_state'
  %139 = GetItem(%135, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_22_decode_layer_call.norm_stats'
  %140 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %137): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_22_decode_layer_call.arg0.distributed_boundary_reshard'
  %141 = Call(Callee: 'decode_layer', %140, %138, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0223', Output: 'layer_22_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0224', Output: 'layer_22_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0225', Output: 'layer_22_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0226', Output: 'layer_22_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0227', Output: 'layer_22_decode_layer_call.arg6.boundary_pack.reshard254'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0228', Output: 'layer_22_decode_layer_call.arg7.boundary_pack.reshard255'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0229', Output: 'layer_22_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0230', Output: 'layer_22_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0232', Output: 'layer_22_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0231', Output: 'layer_22_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 22), const(bool[] : False), %5, %6, %139): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_22_decode_layer_call'
  %142 = GetItem(%141, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_22_decoder_output.boundary_packed.distributed_boundary_abi'
  %143 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %142): // {f32<4>[1,512], (B,B), [1,512], } name='layer_22_decoder_output.boundary_packed'
  %144 = GetItem(%141, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_22_updated_state'
  %145 = GetItem(%141, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_23_decode_layer_call.norm_stats'
  %146 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %143): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_23_decode_layer_call.arg0.distributed_boundary_reshard'
  %147 = Call(Callee: 'decode_layer', %146, %144, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0233', Output: 'layer_23_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0234', Output: 'layer_23_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0235', Output: 'layer_23_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0236', Output: 'layer_23_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0237', Output: 'layer_23_decode_layer_call.arg6.boundary_pack.reshard264'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0238', Output: 'layer_23_decode_layer_call.arg7.boundary_pack.reshard265'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0239', Output: 'layer_23_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0240', Output: 'layer_23_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0242', Output: 'layer_23_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0241', Output: 'layer_23_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 23), const(bool[] : False), %5, %6, %145): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_23_decode_layer_call'
  %148 = GetItem(%147, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_23_decoder_output.boundary_packed.distributed_boundary_abi'
  %149 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %148): // {f32<4>[1,512], (B,B), [1,512], } name='layer_23_decoder_output.boundary_packed'
  %150 = GetItem(%147, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_23_updated_state'
  %151 = GetItem(%147, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_24_decode_layer_call.norm_stats'
  %152 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %149): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_24_decode_layer_call.arg0.distributed_boundary_reshard'
  %153 = Call(Callee: 'decode_layer', %152, %150, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0243', Output: 'layer_24_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0244', Output: 'layer_24_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0245', Output: 'layer_24_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0246', Output: 'layer_24_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0247', Output: 'layer_24_decode_layer_call.arg6.boundary_pack.reshard274'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0248', Output: 'layer_24_decode_layer_call.arg7.boundary_pack.reshard275'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0249', Output: 'layer_24_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0250', Output: 'layer_24_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0252', Output: 'layer_24_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0251', Output: 'layer_24_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 24), const(bool[] : False), %5, %6, %151): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_24_decode_layer_call'
  %154 = GetItem(%153, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_24_decoder_output.boundary_packed.distributed_boundary_abi'
  %155 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %154): // {f32<4>[1,512], (B,B), [1,512], } name='layer_24_decoder_output.boundary_packed'
  %156 = GetItem(%153, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_24_updated_state'
  %157 = GetItem(%153, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_25_decode_layer_call.norm_stats'
  %158 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %155): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_25_decode_layer_call.arg0.distributed_boundary_reshard'
  %159 = Call(Callee: 'decode_layer', %158, %156, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0253', Output: 'layer_25_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0254', Output: 'layer_25_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0255', Output: 'layer_25_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0256', Output: 'layer_25_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0257', Output: 'layer_25_decode_layer_call.arg6.boundary_pack.reshard284'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0258', Output: 'layer_25_decode_layer_call.arg7.boundary_pack.reshard285'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0259', Output: 'layer_25_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0260', Output: 'layer_25_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0262', Output: 'layer_25_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0261', Output: 'layer_25_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 25), const(bool[] : False), %5, %6, %157): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_25_decode_layer_call'
  %160 = GetItem(%159, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_25_decoder_output.boundary_packed.distributed_boundary_abi'
  %161 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %160): // {f32<4>[1,512], (B,B), [1,512], } name='layer_25_decoder_output.boundary_packed'
  %162 = GetItem(%159, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_25_updated_state'
  %163 = GetItem(%159, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_26_decode_layer_call.norm_stats'
  %164 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %161): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_26_decode_layer_call.arg0.distributed_boundary_reshard'
  %165 = Call(Callee: 'decode_layer', %164, %162, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0263', Output: 'layer_26_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0264', Output: 'layer_26_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0265', Output: 'layer_26_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0266', Output: 'layer_26_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0267', Output: 'layer_26_decode_layer_call.arg6.boundary_pack.reshard294'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0268', Output: 'layer_26_decode_layer_call.arg7.boundary_pack.reshard295'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0269', Output: 'layer_26_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0270', Output: 'layer_26_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0272', Output: 'layer_26_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0271', Output: 'layer_26_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 26), const(bool[] : False), %5, %6, %163): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_26_decode_layer_call'
  %166 = GetItem(%165, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_26_decoder_output.boundary_packed.distributed_boundary_abi'
  %167 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %166): // {f32<4>[1,512], (B,B), [1,512], } name='layer_26_decoder_output.boundary_packed'
  %168 = GetItem(%165, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_26_updated_state'
  %169 = GetItem(%165, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='layer_27_decode_layer_call.norm_stats'
  %170 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: , %167): // {f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], } name='layer_27_decode_layer_call.arg0.distributed_boundary_reshard'
  %171 = Call(Callee: 'decode_layer', %170, %168, ConstAssetRef({bf16<4>[512], (S(C(64)@[0])), [64@y], }, Recipe: 'constant_recipe_0273', Output: 'layer_27_decode_layer_call.arg2.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], }, Recipe: 'constant_recipe_0274', Output: 'layer_27_decode_layer_call.arg3.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0275', Output: 'layer_27_decode_layer_call.arg4.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], }, Recipe: 'constant_recipe_0276', Output: 'layer_27_decode_layer_call.arg5.distributed_boundary_reshard'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0277', Output: 'layer_27_decode_layer_call.arg6.boundary_pack.reshard304'), ConstAssetRef({bf16<8>[16], (B), [16], }, Recipe: 'constant_recipe_0278', Output: 'layer_27_decode_layer_call.arg7.boundary_pack.reshard305'), ConstAssetRef({bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], }, Recipe: 'constant_recipe_0279', Output: 'layer_27_decode_layer_call.arg8.distributed_boundary_reshard'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0280', Output: 'layer_27_decode_layer_call.arg9.distributed_boundary_reshard'), ConstAssetRef({bf16[2048,12288], (B,B), [2048,12288], }, Recipe: 'constant_recipe_0282', Output: 'layer_27_decode_layer_call.arg10.distributed_boundary_reshard'), ConstAssetRef({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, Recipe: 'constant_recipe_0281', Output: 'layer_27_decode_layer_call.arg11.distributed_boundary_reshard'), const(i32[] : 27), const(bool[] : True), %5, %6, %169): // ({f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, {f32[1,1,1], (B,B,B), [1,1,1], }) !read_write<paged_attention_kv_cache> name='layer_27_decode_layer_call'
  %172 = GetItem(%171, 0): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='layer_27_decoder_output.boundary_packed.distributed_boundary_abi'
  %173 = GetItem(%171, 2): // {f32[1,1,1], (B,B,B), [1,1,1], } name='final_norm.stats'
  %174 = NN.NormApply(Axis: -1, Epsilon: 1e-06, OutputDtype: 'bfloat16', RoundBeforeScale: False, UseMean: False, %172, %173, ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0283', Output: 'final_norm.vectorized.pack2.reshard311'), ConstAssetRef({bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], }, Recipe: 'constant_recipe_0001', Output: 'decode_layer_input_norm.vectorized.pack3.reshard312')): // {bf16<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='final_norm.vectorized.compute'
  %175 = Distributed.ShardedView(NewType: bf16<4>[1,512], (B,B), [y:8,x:16], Partial: , %174): // {bf16<4>[1,512], (B,B), [1,512], } name='final_norm.vectorized.compute.reshard313'
  %176 = Tensors.Bitcast(Dtype: 'bfloat16', %175): // {bf16[1,2048], (B,B), [1,2048], } name='final_norm'
  %177 = NTT.PackedMatMul(FusedReduce: False, OutputDataType: 'float32', RhsLayout: 'k_major', %176, ConstAssetRef({bf16<8,2,8>[128,18992], (B,S(BC(1)@[0,1])), [128,⌈18992/128⌉@y@x], }, Recipe: 'constant_recipe_0000', Output: 'logits.vectorized.compute.rhs_k_major.physical.reshard314'), %w0, %w0): // {f32<2,4>[1,18992], (B,S(BC(1)@[0,1])), [1,⌈18992/128⌉@y@x], } name='logits.vectorized.compute'
  %178 = Tensors.Bitcast(Dtype: 'float32', %177): // {f32[1,151936], (B,S(BC(8)@[0,1])), [1,1187@y@x], } name='logits'
  %179 = NN.GreedySample(%178): // {i32[1], (B), [1], } name='next_token'
  %180 = GetItem(%171, 1): // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='updated_state'
  %181 = Distributed.ShardedView(NewType: f32[1,151936], (B,B), [y:8,x:16], Partial: , %178): // {f32[1,151936], (B,B), [1,151936], } name='logits.reshard328'
  return (%181, %179, %180)
}

// selections
select vectorization.final_norm = vectorization.norm_apply.reduction_axis // kind=vectorization, origin=default-policy
select vectorization.logits = vectorization.matmul.n // kind=vectorization, origin=default-policy
select packing.lm_head = packing.k_major_n8_k16 // kind=packing, origin=default-policy
select distribution.token_embedding.packed = distribution.token_embedding.packed.embedding_exact_output_sbp.mesh_yx_8x16_bb.in_d_b__d_b_s_c_h0_1_g4.out_d_b_s_c_h0_1_g4 // kind=distribution, origin=agent
select distribution.layer_0_decode_layer_call.arg0.boundary_pack = distribution.layer_0_decode_layer_call.arg0.boundary_pack.operation_type_inference_sbp.mesh_yx_8x16_bb.in_d_b_b.out_d_b_b // kind=distribution, origin=agent
select distribution.layer_0_decode_layer_call.norm_stats = distribution.layer_0_decode_layer_call.norm_stats.norm_stats_additive_sbp.mesh_yx_8x16_bb.in_d_b_b.out_d_b_b_b // kind=distribution, origin=agent
select distribution.final_norm.vectorized.compute = distribution.final_norm.vectorized.compute.norm_apply_compatible_sbp.mesh_yx_8x16_bb.in_d_b_s_c_h0_1_g4__d_b_b_b__d_s_c_h0_1_g4__d_s_c_h0_1_g4.out_d_b_s_c_h0_1_g4 // kind=distribution, origin=agent
select distribution.final_norm = distribution.final_norm.operation_type_inference_sbp.mesh_yx_8x16_bb.in_d_b_b.out_d_b_b // kind=distribution, origin=agent
select distribution.logits.vectorized.compute = distribution.logits.vectorized.compute.packed_matmul_operand_sbp.mesh_yx_8x16_bb.in_d_b_b__d_b_s_bc_h0_1_b1__nonetype__nonetype.out_d_b_s_bc_h0_1_b1 // kind=distribution, origin=agent
select distribution.logits = distribution.logits.operation_type_inference_sbp.mesh_yx_8x16_bb.in_d_b_s_bc_h0_1_b1.out_d_b_s_bc_h0_1_b8 // kind=distribution, origin=agent
select distribution.next_token = distribution.next_token.local_argmax_and_global_index_merge.mesh_yx_8x16_bb.in_d_b_s_bc_h0_1_b8.out_d_b // kind=distribution, origin=agent
select tir.token_embedding.packed = tir.embedding.decode // kind=tir, origin=default-policy
select tir.main_rotary_embedding.packed = semantic.nn.rotary_embedding // kind=semantic_tir, origin=default-policy
select tir.layer_0_decode_layer_call.norm_stats = tir.norm_stats.local // kind=tir, origin=default-policy
select tir.final_norm.vectorized.compute = tir.norm_apply.local // kind=tir, origin=default-policy
select tir.logits.vectorized.compute = tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_tn64_bk512 // kind=tir, origin=default-policy
select tir.next_token = tir.greedy_sample.distributed_argmax // kind=tir, origin=default-policy
