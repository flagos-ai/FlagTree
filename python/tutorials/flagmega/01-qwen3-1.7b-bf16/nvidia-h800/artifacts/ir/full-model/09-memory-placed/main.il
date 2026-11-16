// FlagMega semantic_tir stage=memory_placed_tir semantic_hash=6096fb1c8df78ba180718c3c0935cedabd1894a7cb4fb4ecd39729666b3d1bd0

T.KernelDef("kernel_embedding_decode_9c2800d81a0929bf", kind='triton', %indices: Dist(i32[1], (B), [1], ) [input], %weight: Dist(bf16<4>[151936,512], (B,S(C(4)@[0,1])), [151936,4@y@x], ) [input], %result: Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) [output]) {
  T.Kernel(SemanticOp: 'nn.embedding', SemanticCandidate: 'tir.embedding.decode', MicroKernel: 'tir.embedding.decode', Parameters: {distribution: {input_types: [Dist(i32[1], (B), [y:8,x:16], Partial: ), Dist(bf16<4>[151936,512], (B,S(C(4)@[0,1])), [y:8,x:16], Partial: )], output_type: Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, elements_per_program: 16, family: 'embedding', variant: 'decode'}, Inputs: [%indices, %weight], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [], Reads: ['indices', 'weight'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_nn_rotary_embedding_semantic_nn_rotary_embedding_94f1f085ce07d90d", kind='triton', %reference: Dist(f32<4>[1,512], (B,B), [1,512], ) [metadata], %state: &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} [input], %result_0: Dist(bf16<2,8>[1,1,8], (B,B,B), [1,1,8], ) [output], %result_1: Dist(bf16<2,8>[1,1,8], (B,B,B), [1,1,8], ) [output]) {
  T.Kernel(SemanticOp: 'nn.rotary_embedding', SemanticCandidate: 'semantic.nn.rotary_embedding', MicroKernel: 'tir.rotary_embedding.decode', Parameters: {bounded_accurate_trig: True, distribution: {input_types: [Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}], output_type: (Dist(bf16<2,8>[1,1,8], (B,B,B), [1,1,8], ), Dist(bf16<2,8>[1,1,8], (B,B,B), [1,1,8], )), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, elements_per_program: 128, family: 'rotary_embedding', variant: 'decode'}, Inputs: [%reference, %state], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [], Reads: ['state'], Writes: ['result_0', 'result_1']) -> (%result_0, %result_1)
}

T.KernelDef("kernel_norm_apply_local_990d45338653c5c2", kind='triton', %input: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) [input], %stats: Dist(f32[1,1,1], (B,B,B), [1,1,1], ) [input], %scale: Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) [input], %bias: Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) [input], %result: Dist(bf16<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) [output]) {
  T.Kernel(SemanticOp: 'nn.norm_apply', SemanticCandidate: 'tir.norm_apply.local', MicroKernel: 'tir.norm_apply.local', Parameters: {axis: 1, bias_mode: 'tensor', block_size: 1024, distribution: {input_types: [Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), Dist(f32[1,1,1], (B,B,B), [y:8,x:16], Partial: ), Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: )], output_type: Dist(bf16<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, family: 'norm_apply', use_mean: False, variant: 'local', vector_schedule: {contract: {axes: [1], kind: 'reduction_axis', lane_count: 4, lanes: [4], source_candidate: 'vectorization.norm_apply.reduction_axis'}, lowering: 'local_elementwise', physical: {block_size: 1024}}}, Inputs: [%input, %stats, %scale, %bias], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [%result <- %input], Reads: ['input', 'stats', 'scale', 'bias'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_ntt_packed_qkv_parallel_linear_semantic_ntt_packed_qkv_parallel__e4c2ee4f02f7b3c3", kind='triton', %input: Dist(bf16[1,2048], (B,S(C(256)@[0])), [1,256@y], ) [input], %q_weight_qkv_fused: bf16<8,2,8>[128,16,32] [input], %q_bias: none [input], %k_bias: none [input], %v_bias: none [input], %q_input_scale: none [input], %k_input_scale: none [input], %v_input_scale: none [input], %q_weight_scale: none [input], %k_weight_scale: none [input], %v_weight_scale: none [input], %result_0: Dist(f32<8>[1,256], (B,S(BC(8)@[1])), [1,16@x], P([0], Sum)) [output], %result_1: Dist(f32<8>[1,128], (B,S(BC(8)@[1])), [1,8@x], P([0], Sum)) [output], %result_2: Dist(f32<8>[1,128], (B,S(BC(8)@[1])), [1,8@x], P([0], Sum)) [output]) {
  T.Kernel(SemanticOp: 'ntt.packed_qkv_parallel_linear_fused_rhs', SemanticCandidate: 'semantic.ntt.packed_qkv_parallel_linear', MicroKernel: 'tir.qkv_parallel_linear.packed_partial_mma_smem_pipeline', Parameters: {block_k: 64, block_n: 256, consumer_warps: 8, distribution: {input_types: [Dist(bf16[1,2048], (B,S(C(256)@[0])), [y:8,x:16], Partial: ), bf16<8,2,8>[128,16,32], none, none, none, none, none, none, none, none, none], output_type: (Dist(f32<8>[1,256], (B,S(BC(8)@[1])), [1,16@x], P([0], Sum)), Dist(f32<8>[1,128], (B,S(BC(8)@[1])), [1,8@x], P([0], Sum)), Dist(f32<8>[1,128], (B,S(BC(8)@[1])), [1,8@x], P([0], Sum))), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, family: 'qkv_parallel_linear', num_stages: 2, producer_registers: 24, producer_warps: 1, variant: 'packed_mma_smem_pipeline', worker_width: 32}, Inputs: [%input, %q_weight_qkv_fused, %q_bias, %k_bias, %v_bias, %q_input_scale, %k_input_scale, %v_input_scale, %q_weight_scale, %k_weight_scale, %v_weight_scale], Workspaces: [], SharedWorkspaces: [%rhs_stage: bf16[2,1,32,4,2,64]@shared, %lhs_stage: bf16[1,256]@shared], InplaceAliases: [], Reads: ['input', 'q_weight_qkv_fused', 'q_bias', 'k_bias', 'v_bias', 'q_input_scale', 'k_input_scale', 'v_input_scale', 'q_weight_scale', 'k_weight_scale', 'v_weight_scale'], Writes: ['result_0', 'result_1', 'result_2']) -> (%result_0, %result_1, %result_2)
}

T.KernelDef("kernel_ntt_gather_reduce_qkv_rope_with_cache_semantic_ntt_gather_reduce_4de2b227526601fc", kind='triton', %qkv: (Dist(f32<8>[1,256], (B,S(BC(8)@[1])), [1,16@x], P([0], Sum)), Dist(f32<8>[1,128], (B,S(BC(8)@[1])), [1,8@x], P([0], Sum)), Dist(f32<8>[1,128], (B,S(BC(8)@[1])), [1,8@x], P([0], Sum))) [input], %q_scale: Dist(bf16<8>[16], (B), [16], ) [input], %k_scale: Dist(bf16<8>[16], (B), [16], ) [input], %q_bias: Dist(bf16<8>[16], (B), [16], ) [input], %k_bias: Dist(bf16<8>[16], (B), [16], ) [input], %cos: Dist(bf16<2,8>[1,1,8], (B,B,B), [1,1,8], ) [input], %sin: Dist(bf16<2,8>[1,1,8], (B,B,B), [1,1,8], ) [input], %state: &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} [inout], %layer_id: i32[] [input], %advance_sequence: bool[] [input], %result_0: Dist(bf16<8>[1,16,16], (B,S(C(2)@[0]),B), [1,2@y,16], ) [output], %result_1: &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} [output]) {
  T.Kernel(SemanticOp: 'ntt.gather_reduce_qkv_rope_with_cache', SemanticCandidate: 'semantic.ntt.gather_reduce_qkv_rope_with_cache', MicroKernel: 'tir.gather_reduce_qkv_rope_with_cache.decode', Parameters: {distribution: {input_types: [(Dist(f32<8>[1,256], (B,S(BC(8)@[1])), [1,16@x], P([0], Sum)), Dist(f32<8>[1,128], (B,S(BC(8)@[1])), [1,8@x], P([0], Sum)), Dist(f32<8>[1,128], (B,S(BC(8)@[1])), [1,8@x], P([0], Sum))), Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), Dist(bf16<2,8>[1,1,8], (B,B,B), [y:8,x:16], Partial: ), Dist(bf16<2,8>[1,1,8], (B,B,B), [y:8,x:16], Partial: ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, i32[], bool[]], output_type: (Dist(bf16<8>[1,16,16], (B,S(C(2)@[0]),B), [1,2@y,16], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, elements_per_program: 128, family: 'gather_reduce_qkv_rope_with_cache', variant: 'decode'}, Inputs: [%qkv, %q_scale, %k_scale, %q_bias, %k_bias, %cos, %sin, %state, %layer_id, %advance_sequence], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [], Reads: ['qkv', 'q_scale', 'k_scale', 'q_bias', 'k_bias', 'cos', 'sin', 'state'], Writes: ['state', 'result_0']) -> (%result_0, %result_1)
}

T.KernelDef("kernel_ntt_paged_attention_partial_semantic_ntt_paged_attention_partial_499e606b2b12cd57", kind='triton', %q: Dist(bf16<8>[1,16,16], (B,S(C(2)@[0]),B), [1,2@y,16], ) [input], %state: &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} [input], %layer_id: i32[] [input], %result_0: Dist(f32[1,16,1], (B,S(C(2)@[0]),B), [1,2@y,1], P([1], Max)) [output], %result_1: Dist(f32[1,16,1], (B,S(C(2)@[0]),B), [1,2@y,1], P([1], Sum)) [output], %result_2: Dist(f32[1,16,128], (B,S(C(2)@[0]),B), [1,2@y,128], P([1], Sum)) [output]) {
  T.Kernel(SemanticOp: 'ntt.paged_attention_partial', SemanticCandidate: 'semantic.ntt.paged_attention_partial', MicroKernel: 'tir.paged_attention_partial.mma_tma_smem_pipeline', Parameters: {block_k: 128, block_m: 1, block_n: 64, consumer_warps: 8, distribution: {input_types: [Dist(bf16<8>[1,16,16], (B,S(C(2)@[0]),B), [y:8,x:16], Partial: ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, i32[]], output_type: (Dist(f32[1,16,1], (B,S(C(2)@[0]),B), [1,2@y,1], P([1], Max)), Dist(f32[1,16,1], (B,S(C(2)@[0]),B), [1,2@y,1], P([1], Sum)), Dist(f32[1,16,128], (B,S(C(2)@[0]),B), [1,2@y,128], P([1], Sum))), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, family: 'paged_attention_partial', num_stages: 2, producer_registers: 24, producer_warps: 1, variant: 'mma_tma_smem_pipeline', worker_width: 32}, Inputs: [%q, %state, %layer_id], Workspaces: [], SharedWorkspaces: [%key_stage: bf16[2,1,1,64,1,128]@shared, %value_stage: bf16[2,1,1,64,1,128]@shared], InplaceAliases: [], Reads: ['q', 'state'], Writes: ['result_0', 'result_1', 'result_2']) -> (%result_0, %result_1, %result_2)
}

T.KernelDef("kernel_ntt_paged_attention_combine_semantic_ntt_paged_attention_combine_9503a711940aa3e2", kind='triton', %max_state: Dist(f32[1,16,1], (B,S(C(2)@[0]),B), [1,2@y,1], P([1], Max)) [input], %sum_state: Dist(f32[1,16,1], (B,S(C(2)@[0]),B), [1,2@y,1], P([1], Sum)) [input], %acc_state: Dist(f32[1,16,128], (B,S(C(2)@[0]),B), [1,2@y,128], P([1], Sum)) [input], %result: Dist(bf16<8>[1,16,16], (B,S(C(2)@[0]),B), [1,2@y,16], ) [output]) {
  T.Kernel(SemanticOp: 'ntt.paged_attention_combine', SemanticCandidate: 'semantic.ntt.paged_attention_combine', MicroKernel: 'tir.paged_attention_combine.decode', Parameters: {distribution: {input_types: [Dist(f32[1,16,1], (B,S(C(2)@[0]),B), [y:8,x:16], Partial: P([1], Max)), Dist(f32[1,16,1], (B,S(C(2)@[0]),B), [y:8,x:16], Partial: P([1], Sum)), Dist(f32[1,16,128], (B,S(C(2)@[0]),B), [y:8,x:16], Partial: P([1], Sum))], output_type: Dist(bf16<8>[1,16,16], (B,S(C(2)@[0]),B), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, elements_per_program: 128, family: 'paged_attention_combine', variant: 'decode'}, Inputs: [%max_state, %sum_state, %acc_state], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [], Reads: ['max_state', 'sum_state', 'acc_state'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_norm_stats_local_6f038d0656f1ea04", kind='triton', %input: Dist(f32<4>[1,512], (B,B), [1,512], ) [input], %result: Dist(f32[1,1,1], (B,B,B), [1,1,1], ) [output]) {
  T.Kernel(SemanticOp: 'nn.norm_stats', SemanticCandidate: 'tir.norm_stats.local', MicroKernel: 'tir.norm_stats.local', Parameters: {axis: 1, block_size: 128, distribution: {input_types: [Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: )], output_type: Dist(f32[1,1,1], (B,B,B), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, family: 'norm_stats', use_mean: False, variant: 'local'}, Inputs: [%input], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [], Reads: ['input'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_norm_apply_local_f33fb3dd3b151d1a", kind='triton', %input: Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) [input], %stats: Dist(f32[1,1,1], (B,B,B), [1,1,1], ) [input], %scale: Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) [input], %bias: Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) [input], %result: Dist(bf16<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) [output]) {
  T.Kernel(SemanticOp: 'nn.norm_apply', SemanticCandidate: 'tir.norm_apply.local', MicroKernel: 'tir.norm_apply.local', Parameters: {axis: 1, bias_mode: 'tensor', block_size: 1024, distribution: {input_types: [Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [y:8,x:16], Partial: ), Dist(f32[1,1,1], (B,B,B), [y:8,x:16], Partial: ), Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: )], output_type: Dist(bf16<4>[1,512], (B,S(C(4)@[0,1])), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, family: 'norm_apply', use_mean: False, variant: 'local', vector_schedule: {contract: {axes: [1], kind: 'reduction_axis', lane_count: 4, lanes: [4], source_candidate: 'vectorization.norm_apply.reduction_axis'}, lowering: 'local_elementwise', physical: {block_size: 1024}}}, Inputs: [%input, %stats, %scale, %bias], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [%result <- %input], Reads: ['input', 'stats', 'scale', 'bias'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_dense_matmul_packed_tensor_descriptor_smem_pipeline_gemv_f25069adf84b6d9b", kind='triton', %lhs: Dist(bf16[1,2048], (B,B), [1,2048], ) [input], %rhs: Dist(bf16<8,2,8>[128,18992], (B,S(BC(1)@[0,1])), [128,⌈18992/128⌉@y@x], ) [input], %scale: none [input], %addend: none [input], %result: Dist(f32<2,4>[1,18992], (B,S(BC(1)@[0,1])), [1,⌈18992/128⌉@y@x], ) [output]) {
  T.Kernel(SemanticOp: 'ntt.packed_matmul', SemanticCandidate: 'tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_tn64_bk512', MicroKernel: 'tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_tn64_bk512', Parameters: {block_k: 512, consumer_warps: 8, descriptor_kind: 'table', distribution: {input_types: [Dist(bf16[1,2048], (B,B), [y:8,x:16], Partial: ), Dist(bf16<8,2,8>[128,18992], (B,S(BC(1)@[0,1])), [y:8,x:16], Partial: ), none, none], output_type: Dist(f32<2,4>[1,18992], (B,S(BC(1)@[0,1])), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, distribution_schedule: {kind: 'canonical'}, family: 'dense_matmul', num_stages: 3, packed_layout: 'k_major_n8_k16', producer_registers: 24, producer_warps: 1, reduction_group: 32, tile_n: 64, variant: 'packed_tensor_descriptor_smem_pipeline_gemv', vector_schedule: {contract: {axes: [1, 1], kind: 'output_axis', lane_count: 8, lanes: [2, 4], source_candidate: 'vectorization.matmul.n'}, lowering: 'output_tile', physical: {tile_n: 64}}, worker_width: 32}, Inputs: [%lhs, %rhs, %scale, %addend], Workspaces: [], SharedWorkspaces: [%weight_stage: bf16[2,32,8,2,64]@shared], InplaceAliases: [], Reads: ['lhs', 'rhs', 'scale', 'addend'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_greedy_sample_distributed_argmax_65438e715cd66e9c", kind='triton', %logits: Dist(f32[1,151936], (B,S(BC(8)@[0,1])), [1,1187@y@x], ) [input], %result: Dist(i32[1], (B), [1], ) [output], %partial_max: f32[1,128] [workspace], %partial_index: i32[1,128] [workspace]) {
  T.Kernel(SemanticOp: 'nn.greedy_sample', SemanticCandidate: 'tir.greedy_sample.distributed_argmax', MicroKernel: 'tir.greedy_sample.distributed_argmax', Parameters: {distribution: {input_types: [Dist(f32[1,151936], (B,S(BC(8)@[0,1])), [y:8,x:16], Partial: )], output_type: Dist(i32[1], (B), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, family: 'greedy_sample', tie_break: 'lowest_index', variant: 'distributed_argmax', vocab_tile: 1024}, Inputs: [%logits], Workspaces: [%partial_max: f32[1,128]@workspace, %partial_index: i32[1,128]@workspace], SharedWorkspaces: [], InplaceAliases: [], Reads: ['logits'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_dense_matmul_packed_tensor_descriptor_smem_pipeline_gemv_83f59f3c2c220412", kind='triton', %lhs: Dist(bf16[1,2048], (B,B), [1,2048], ) [input], %rhs: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) [input], %scale: none [input], %addend: none [input], %result: Dist(bf16<8>[1,256], (B,S(C(2)@[0,1])), [1,2@y@x], ) [output]) {
  T.Kernel(SemanticOp: 'ntt.packed_matmul', SemanticCandidate: 'tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv', MicroKernel: 'tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv', Parameters: {block_k: 1024, consumer_warps: 8, distribution: {input_types: [Dist(bf16[1,2048], (B,B), [y:8,x:16], Partial: ), Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), none, none], output_type: Dist(bf16<8>[1,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, distribution_schedule: {kind: 'canonical'}, family: 'dense_matmul', num_stages: 3, packed_layout: 'k_major_n8_k16', producer_registers: 24, producer_warps: 1, reduction_group: 32, tile_n: 16, variant: 'packed_tensor_descriptor_smem_pipeline_gemv', vector_schedule: {contract: {axes: [1], kind: 'output_axis', lane_count: 8, lanes: [8], source_candidate: 'vectorization.matmul.n'}, lowering: 'output_tile', physical: {tile_n: 16}}, worker_width: 32}, Inputs: [%lhs, %rhs, %scale, %addend], Workspaces: [], SharedWorkspaces: [%weight_stage: bf16[4,64,2,2,64]@shared], InplaceAliases: [], Reads: ['lhs', 'rhs', 'scale', 'addend'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_elementwise_add_e4db529109683098", kind='triton', %lhs: Dist(f32<2,2>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) [input], %rhs: Dist(bf16[1,2048], (B,S(C(16)@[0,1])), [1,16@y@x], ) [input], %result: Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) [output]) {
  T.Kernel(SemanticOp: 'math.vectorized_binary', SemanticCandidate: 'tir.elementwise.add', MicroKernel: 'tir.elementwise.add', Parameters: {distribution: {input_types: [Dist(f32<2,2>[1,512], (B,S(C(4)@[0,1])), [y:8,x:16], Partial: ), Dist(bf16[1,2048], (B,S(C(16)@[0,1])), [y:8,x:16], Partial: )], output_type: Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, family: 'elementwise', variant: 'add', vector_groups: 32, vector_schedule: {contract: {axes: [1], kind: 'axes', lane_count: 4, lanes: [4], source_candidate: 'fused-typed-ir'}, lowering: 'packed_axes', physical: {elements_per_program: 128}}}, Inputs: [%lhs, %rhs], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [], Reads: ['lhs', 'rhs'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_norm_stats_local_f3ea5eb113c32dbc", kind='triton', %input: Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) [input], %result: Dist(f32[1,1,1], (B,B,B), [1,1,1], P([0,1], Sum)) [output]) {
  T.Kernel(SemanticOp: 'nn.norm_stats', SemanticCandidate: 'tir.norm_stats.local', MicroKernel: 'tir.norm_stats.local', Parameters: {axis: 1, block_size: 128, distribution: {input_types: [Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [y:8,x:16], Partial: )], output_type: Dist(f32[1,1,1], (B,B,B), [y:8,x:16], Partial: P([0,1], Sum)), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, family: 'norm_stats', use_mean: False, variant: 'local'}, Inputs: [%input], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [], Reads: ['input'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_distributed_boxing_gather_reduce_scatter_c7081da1dd8cd347", kind='triton', %value: Dist(f32[1,1,1], (B,B,B), [1,1,1], P([0,1], Sum)) [input], %result: Dist(f32[1,1,1], (B,B,B), [1,1,1], ) [output]) {
  T.Kernel(SemanticOp: 'distributed.boxing', SemanticCandidate: 'tir.distributed_boxing.gather_reduce_scatter', MicroKernel: 'tir.distributed_boxing.gather_reduce_scatter', Parameters: {distribution: {input_types: [Dist(f32[1,1,1], (B,B,B), [y:8,x:16], Partial: P([0,1], Sum))], output_type: Dist(f32[1,1,1], (B,B,B), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, family: 'distributed_boxing', leaf_transitions: ['gather_reduce_scatter'], source_type: Dist(f32[1,1,1], (B,B,B), [y:8,x:16], Partial: P([0,1], Sum)), target_type: Dist(f32[1,1,1], (B,B,B), [y:8,x:16], Partial: ), tile: 1024, variant: 'gather_reduce_scatter'}, Inputs: [%value], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [], Reads: ['value'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_split_k_dense_glu_projection_795e494d3d3d1f6f", kind='triton', %value: Dist(bf16[1,2048], (B,B), [1,2048], ) [input], %weight: Dist(bf16[2048,12288], (B,B), [2048,12288], ) [input], %result: Dist(f32[2,12288], (S(C@[1]),S(C@[0])), [1@x,256@y], ) [output]) {
  T.Kernel(SemanticOp: 'ntt.split_k_dense_glu_projection', SemanticCandidate: 'tir.split_k_dense_glu_projection.participant_retire', MicroKernel: 'tir.split_k_dense_glu_projection.participant_retire', Parameters: {block_k: 64, compute_num_warps: 8, distribution: {input_types: [Dist(bf16[1,2048], (B,B), [y:48,x:2], Partial: ), Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: )], output_type: Dist(f32[2,12288], (S(C@[1]),S(C@[0])), [y:48,x:2], Partial: ), placement: {hierarchy: [48, 2], hierarchy_levels: 'bb', name: 'yx'}}, family: 'split_k_dense_glu', maxnreg: 64, num_stages: 3, program_grid: [48, 2, 1], program_owner_mapping: 'x-fast', split_k: 2, tile_n: 256, variant: 'projection'}, Inputs: [%value, %weight], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [], Reads: ['value', 'weight'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_split_k_dense_glu_epilogue_8df8ded3189c5058", kind='triton', %partial: Dist(f32[2,12288], (B,B), [2,12288], ) [input], %result: Dist(bf16[1,6144], (B,S(C@[0])), [1,256@a], ) [output]) {
  T.Kernel(SemanticOp: 'ntt.split_k_dense_glu_epilogue', SemanticCandidate: 'tir.split_k_dense_glu_epilogue.complete_sum', MicroKernel: 'tir.split_k_dense_glu_epilogue.complete_sum', Parameters: {compute_num_warps: 8, distribution: {input_types: [Dist(f32[2,12288], (B,B), [a:24], Partial: )], output_type: Dist(bf16[1,6144], (B,S(C@[0])), [a:24], Partial: ), placement: {hierarchy: [24], hierarchy_levels: 'b', name: 'a'}}, family: 'split_k_dense_glu', program_grid: [24, 1, 1], program_owner_mapping: 'x-fast', tile_n: 256, variant: 'epilogue'}, Inputs: [%partial], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [], Reads: ['partial'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_dense_matmul_packed_tensor_descriptor_smem_pipeline_gemv_3bc638f6708ca217", kind='triton', %lhs: Dist(bf16[1,6144], (B,B), [1,6144], ) [input], %rhs: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) [input], %scale: none [input], %addend: none [input], %result: Dist(bf16<8>[1,256], (B,S(C(2)@[0,1])), [1,2@y@x], ) [output]) {
  T.Kernel(SemanticOp: 'ntt.packed_matmul', SemanticCandidate: 'tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv_lhs8192_async', MicroKernel: 'tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv_lhs8192_async', Parameters: {block_k: 1024, consumer_warps: 8, distribution: {input_types: [Dist(bf16[1,6144], (B,B), [y:8,x:16], Partial: ), Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), none, none], output_type: Dist(bf16<8>[1,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, distribution_schedule: {kind: 'canonical'}, family: 'dense_matmul', lhs_copy_kind: 'async', lhs_copy_tile: 1024, lhs_stage_extent: 8192, num_stages: 3, packed_layout: 'k_major_n8_k16', producer_registers: 24, producer_warps: 1, reduction_group: 32, tile_n: 16, variant: 'packed_tensor_descriptor_smem_pipeline_gemv', vector_schedule: {contract: {axes: [1], kind: 'output_axis', lane_count: 8, lanes: [8], source_candidate: 'vectorization.matmul.n'}, lowering: 'output_tile', physical: {tile_n: 16}}, worker_width: 32}, Inputs: [%lhs, %rhs, %scale, %addend], Workspaces: [], SharedWorkspaces: [%weight_stage: bf16[4,64,2,2,64]@shared, %lhs_stage: bf16[1,8192]@shared], InplaceAliases: [], Reads: ['lhs', 'rhs', 'scale', 'addend'], Writes: ['result']) -> (%result)
}

T.KernelDef("kernel_elementwise_add_c14d1af7ce938a1b", kind='triton', %lhs: Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) [input], %rhs: Dist(bf16[1,2048], (B,S(C(16)@[0,1])), [1,16@y@x], ) [input], %result: Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) [output]) {
  T.Kernel(SemanticOp: 'math.vectorized_binary', SemanticCandidate: 'tir.elementwise.add', MicroKernel: 'tir.elementwise.add', Parameters: {distribution: {input_types: [Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [y:8,x:16], Partial: ), Dist(bf16[1,2048], (B,S(C(16)@[0,1])), [y:8,x:16], Partial: )], output_type: Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [y:8,x:16], Partial: ), placement: {hierarchy: [8, 16], hierarchy_levels: 'bb', name: 'yx'}}, family: 'elementwise', variant: 'add', vector_groups: 32, vector_schedule: {contract: {axes: [1], kind: 'axes', lane_count: 4, lanes: [4], source_candidate: 'vectorization.last_axis'}, lowering: 'packed_axes', physical: {elements_per_program: 128}}}, Inputs: [%lhs, %rhs], Workspaces: [], SharedWorkspaces: [], InplaceAliases: [], Reads: ['lhs', 'rhs'], Writes: ['result']) -> (%result)
}
T.PrimFunc("main", %input_ids: i32[1], %paged_attention_kv_cache: &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}).Body(
  weights {
    // display grouping only; not an execution schedule
    T.ConstantRecipe constant_recipe_0000 // fingerprint=7ff6b5a5c7bb83a512ede7e7bf834450fbf3a95cfdd32efbd71a7868eb2d776b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [4], T.WeightRef(bf16[151936,2048], 'embed_tokens.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.embed_tokens.weight')): // bf16<4>[151936,512] name='token_embedding.packed_weight'
      %1 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[151936,2048], 'embed_tokens.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.embed_tokens.weight')): // bf16<8>[151936,256] name='logits.vectorized.compute.rhs_k_major.k_vector'
      %2 = Tensors.Pack(Axes: [1], Lanes: [2], %1): // bf16<2,8>[151936,128] name='logits.vectorized.compute.rhs_k_major.k_pack'
      %3 = Tensors.Pack(Axes: [0], Lanes: [8], %2): // bf16<8,2,8>[18992,128] name='logits.vectorized.compute.rhs_k_major.n_vector'
      %4 = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[128,18992] name='logits.vectorized.compute.rhs_k_major.physical'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<4>[151936,512], (B,S(C(4)@[0,1])), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[151936,512], (B,S(C(4)@[0,1])), [151936,4@y@x], ) name='token_embedding.packed_weight.reshard16'
      %6 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,18992], (B,S(BC(1)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,18992], (B,S(BC(1)@[0,1])), [128,⌈18992/128⌉@y@x], ) name='logits.vectorized.compute.rhs_k_major.physical.reshard314'
      yield (%5, %6)
    }
    T.ConstantRecipe constant_recipe_0001 // fingerprint=cbde7c9c1a4a6e7cc1ed09517dc8a6516902c8e1103c8b48a0051788cb5e3db9
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.Const(bf16[2048] : splat(0.0))): // bf16<4>[512] name='decode_layer_input_norm.vectorized.pack3'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='decode_layer_input_norm.vectorized.pack3.reshard21'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='decode_layer_input_norm.vectorized.pack3.reshard312'
      yield (%1, %2)
    }
    T.ConstantRecipe constant_recipe_0003 // fingerprint=c696a7844c662a49e246424aac63aac2f299d053bdd5376082c731698b77f530
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.0.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_0_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_0_decode_layer_call.arg6.boundary_pack.reshard34'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0004 // fingerprint=f1b6231c20be3bdb2813ca6b4d70c4e4b28194a388c5cb25ee0ad871c9166cbb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.0.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_0_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_0_decode_layer_call.arg7.boundary_pack.reshard35'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0005 // fingerprint=d66c64d2b29923ff26165d0b82865bb203a06e395a4913baa838616ff6e48103
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.0.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.input_layernorm.weight')): // bf16<4>[512] name='layer_0_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_0_decode_layer_call.arg2.boundary_pack.reshard30'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_0_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0006 // fingerprint=4478dac65d231d01a2a7fe8b955d127e9bccb7e4d86b37b596ab36328cd7007a
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.0.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_0_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard31'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_0_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.0.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_0_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard32'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_0_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.0.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_0_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard33'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_0_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_0_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_0_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_0_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_0_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0007 // fingerprint=9ddb4b6f6e01c16d0ab77b30c4960cd63837c1586857f8c90d6212f391d2581d
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.0.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_0_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard36'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_0_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0008 // fingerprint=bcbc36fe61ea9f0ff8e418baf361d41b794d30bf481b781cf342a759164809ab
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.0.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_0_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_0_decode_layer_call.arg9.boundary_pack.reshard37'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_0_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0009 // fingerprint=a799feaeb5fe2a74cad677bbde879888d1a9b89a4c023b08510d4b698c1ad8f0
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.0.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.0.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_0_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_0_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_0_decode_layer_call.gu_weight_rowmajor.reshard38'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_0_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0010 // fingerprint=7f432e7446111b7f88635cacd519b646ada15f18c8630c7529413d822efe7009
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.0.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.0.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_0_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard39'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_0_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0011 // fingerprint=3a44e9a51d06633db2298b1d82845e9bfbd2fc4c3e705c57b3a370cbc2759525
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.1.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_1_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_1_decode_layer_call.arg6.boundary_pack.reshard44'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0012 // fingerprint=60bb1c1f1bc6e5ab7f8ded53d1c36952006bccea91e3b517098759432e7cb12c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.1.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_1_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_1_decode_layer_call.arg7.boundary_pack.reshard45'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0013 // fingerprint=eed40ce9818f2959f06a4061864fc422507047b255ef6a0b5a3919dd0dc0b343
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.1.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.input_layernorm.weight')): // bf16<4>[512] name='layer_1_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_1_decode_layer_call.arg2.boundary_pack.reshard40'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_1_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0014 // fingerprint=ff5eae123b80fc5df9fc9c353d220f570f814be11272ce0e8e75b190a4e2868a
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.1.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_1_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard41'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_1_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.1.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_1_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard42'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_1_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.1.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_1_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard43'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_1_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_1_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_1_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_1_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_1_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0015 // fingerprint=8d887e289e070ac57b139e1a03317d941543088a6a32b0b7a71b954439b37c6f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.1.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_1_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard46'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_1_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0016 // fingerprint=759fd7c1bd324769375a106b5d0b739a9c297bda5aefe2e20cb355085aa716da
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.1.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_1_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_1_decode_layer_call.arg9.boundary_pack.reshard47'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_1_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0017 // fingerprint=3b3be02493ac287d7d36380f5785996ae7d2cc716e8fa9fd44c36cdacfb36283
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.1.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.1.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_1_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_1_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_1_decode_layer_call.gu_weight_rowmajor.reshard48'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_1_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0018 // fingerprint=5cd0c09c2eb2d65b7bff78e99adf239a647235beac4426d4e93166a82f9854b3
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.1.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.1.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_1_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard49'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_1_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0019 // fingerprint=319a8f036df20acaf57e160eac7a31044c3e7df7300a70050c082b332f88a3b3
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.2.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_2_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_2_decode_layer_call.arg6.boundary_pack.reshard54'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0020 // fingerprint=20915c6ffdc447c9d77e3fdff2cb5982302c18603238df667a96452bca702a1c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.2.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_2_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_2_decode_layer_call.arg7.boundary_pack.reshard55'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0021 // fingerprint=079a3c7be0b0019d72592c346d14a5b7270635e87fc60ea4c31daa20dfab3fbd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.2.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.input_layernorm.weight')): // bf16<4>[512] name='layer_2_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_2_decode_layer_call.arg2.boundary_pack.reshard50'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_2_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0022 // fingerprint=f556e951855f98498271ddea313175452a8bb5337ad7a1d8df5e92f4fa5c3acd
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.2.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_2_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard51'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_2_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.2.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_2_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard52'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_2_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.2.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_2_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard53'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_2_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_2_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_2_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_2_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_2_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0023 // fingerprint=4c9b8e34b5e73fb46e0356a03649c3d56e518252abd795277aa66574247b3162
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.2.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_2_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard56'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_2_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0024 // fingerprint=cfc5bf57cc32711244c1de290ebee98fa816e06e973c628077bd8ded18d91260
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.2.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_2_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_2_decode_layer_call.arg9.boundary_pack.reshard57'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_2_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0025 // fingerprint=b7bb0c3efb7dbd0caea208999fcd720f28c47dbe1568a6964fc4e1ea4fcba82b
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.2.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.2.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_2_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_2_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_2_decode_layer_call.gu_weight_rowmajor.reshard58'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_2_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0026 // fingerprint=974bfa183c5ea948c936a776d3260ab80ae7e7b0f603a722935836f5802e15f7
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.2.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.2.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_2_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard59'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_2_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0027 // fingerprint=3a1b42604dc83f0c701de6701ea666935d841f0c93b8eb15d654f0d96c030de9
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.3.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_3_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_3_decode_layer_call.arg6.boundary_pack.reshard64'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0028 // fingerprint=b6d2618dce28c3bd141611ff6862756e8e1caee93cacf1158870d0f35a8eee73
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.3.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_3_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_3_decode_layer_call.arg7.boundary_pack.reshard65'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0029 // fingerprint=f49e4daeb5cbe1f8af0865c5f5d0defc847dada292a8e1f1c0306dfbb761b4dd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.3.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.input_layernorm.weight')): // bf16<4>[512] name='layer_3_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_3_decode_layer_call.arg2.boundary_pack.reshard60'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_3_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0030 // fingerprint=a177751d05fdf32f1299bc6cc020391f39aef356b4e421bc031289311c36bfaf
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.3.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_3_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard61'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_3_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.3.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_3_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard62'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_3_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.3.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_3_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard63'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_3_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_3_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_3_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_3_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_3_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0031 // fingerprint=6f7d25fb98065cd089aec7e804e7c21fc45bab2cf07aa615f212e8f1dcc8410e
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.3.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_3_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard66'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_3_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0032 // fingerprint=5a1f4d9b46379cfa3291ef75b46d0a47622fd4d5bdfb0afa9da77db50acd7a5f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.3.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_3_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_3_decode_layer_call.arg9.boundary_pack.reshard67'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_3_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0033 // fingerprint=6d4220bc7b426b992bf4b81f4dd7812a42e2a6d06659f95c04cb5f16bc44e016
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.3.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.3.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_3_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_3_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_3_decode_layer_call.gu_weight_rowmajor.reshard68'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_3_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0034 // fingerprint=8a2be394a568e6ddfcdd7f3430ee750c64ac288ec748d4855219475575ecf113
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.3.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.3.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_3_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard69'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_3_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0035 // fingerprint=4fa9b6555eaf4488e768adc57fa99c079893d76ed14e872444e65d671c9d102a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.4.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_4_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_4_decode_layer_call.arg6.boundary_pack.reshard74'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0036 // fingerprint=8fabd49cf2649b8c0ca0c9fd3b8609fe427a69eeea1593ea64daa10aa56c52a0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.4.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_4_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_4_decode_layer_call.arg7.boundary_pack.reshard75'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0037 // fingerprint=0adafb482c45374533e7eef98dc25cbeccd84efa1e7e60cdcb991cfed923e881
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.4.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.input_layernorm.weight')): // bf16<4>[512] name='layer_4_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_4_decode_layer_call.arg2.boundary_pack.reshard70'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_4_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0038 // fingerprint=aa8cd676da3e0bc0a32c926bed6903948e775f3e0b6adc2e2db9baf23183a8d3
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.4.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_4_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard71'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_4_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.4.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_4_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard72'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_4_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.4.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_4_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard73'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_4_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_4_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_4_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_4_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_4_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0039 // fingerprint=25e55cbb37072856da68bcebeafab501b76b9b59733d6793d68ae7d28b212c58
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.4.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_4_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard76'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_4_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0040 // fingerprint=a71a5de518aa178ad22830e87dfec6bb3b65f6bb15ce7981219a932b79941f79
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.4.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_4_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_4_decode_layer_call.arg9.boundary_pack.reshard77'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_4_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0041 // fingerprint=18bd45538b25fe396b2371ec17c1ca75891778ab2061aa602370bfdb40289abb
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.4.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.4.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_4_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_4_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_4_decode_layer_call.gu_weight_rowmajor.reshard78'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_4_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0042 // fingerprint=4fb1b91de95c82a9165983b1d1f229cb34a73971f05b4a93dff628bfcb87cc74
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.4.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.4.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_4_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard79'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_4_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0043 // fingerprint=245385bb6126ae8e1b6f547cbf8d359d8979f34524c0fd48b1b205f68a173234
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.5.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_5_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_5_decode_layer_call.arg6.boundary_pack.reshard84'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0044 // fingerprint=794d9db54bbe81feb677d0e280aeec8c7e94aa89f7fa281f3b97b1328fa54887
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.5.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_5_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_5_decode_layer_call.arg7.boundary_pack.reshard85'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0045 // fingerprint=a220bda13daa24b99c06c250d028746070feaa6757b1e53ad6967bd9dc11508f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.5.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.input_layernorm.weight')): // bf16<4>[512] name='layer_5_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_5_decode_layer_call.arg2.boundary_pack.reshard80'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_5_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0046 // fingerprint=88faadb235827e1ce2d05836101b0d04afac1fc8968a8859f0ad3553b58badff
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.5.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_5_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard81'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_5_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.5.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_5_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard82'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_5_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.5.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_5_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard83'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_5_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_5_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_5_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_5_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_5_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0047 // fingerprint=027d66e919a9fb2217dfc93138098974f7c5b3065a66253b55aac8c13f461e80
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.5.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_5_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard86'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_5_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0048 // fingerprint=ffffce01315fd3203d4298947a6e8f87a5ca5e8b1290d4b6160ce62d0a14261a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.5.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_5_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_5_decode_layer_call.arg9.boundary_pack.reshard87'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_5_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0049 // fingerprint=b02fd98537c4848e39470a7632b6d40b722b1013824fbeee9c25dc351d46e04c
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.5.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.5.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_5_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_5_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_5_decode_layer_call.gu_weight_rowmajor.reshard88'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_5_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0050 // fingerprint=87c373fae203c7a8f4c75774d83ebd408e2b98a5a846357abc00cf5ffa3e18e9
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.5.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.5.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_5_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard89'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_5_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0051 // fingerprint=70cc78c062c80298e3e4be09a5c1c44e94344f9c6700e8976f0ccd1512c163a1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.6.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_6_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_6_decode_layer_call.arg6.boundary_pack.reshard94'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0052 // fingerprint=1065627638d13d49fd49d641108bc3f35735a0ac0ef5bf99ee5099fcefc9df41
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.6.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_6_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_6_decode_layer_call.arg7.boundary_pack.reshard95'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0053 // fingerprint=d2424ac5e56aefb35b04a7378da9afbfd195010d08d968bdfbb537e8f15b4cf6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.6.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.input_layernorm.weight')): // bf16<4>[512] name='layer_6_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_6_decode_layer_call.arg2.boundary_pack.reshard90'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_6_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0054 // fingerprint=f2188bfa59673eb63522828fc539d6c63985c3e97b868828098183f7610c8533
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.6.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_6_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard91'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_6_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.6.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_6_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard92'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_6_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.6.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_6_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard93'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_6_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_6_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_6_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_6_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_6_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0055 // fingerprint=c0a1185093ea1bd3446cdc8c62d6c82da7e1e3c91782a012246ea853bc00200f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.6.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_6_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard96'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_6_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0056 // fingerprint=574b9fef64290957d8251d3eac02fa4616a682bc539d734a5fa7946be407959f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.6.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_6_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_6_decode_layer_call.arg9.boundary_pack.reshard97'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_6_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0057 // fingerprint=f78b0a43aa44e1befb209791bf0dc5d2037cbc8da4ae8bd0f611830a2150f9fc
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.6.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.6.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_6_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_6_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_6_decode_layer_call.gu_weight_rowmajor.reshard98'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_6_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0058 // fingerprint=a15346bb7928eba7889cfc9fc491cef4d134bb10b98946b6ef3efdaacfe14097
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.6.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.6.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_6_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard99'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_6_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0059 // fingerprint=1f79b638c8d41dbcedaf9d403dbd77094cd27b04a01c40fe0bfcf6ed564a7bd0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.7.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_7_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_7_decode_layer_call.arg6.boundary_pack.reshard104'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0060 // fingerprint=a62790f63f13e96770185a9615ef95924cbaab08210633d680dd1db9adcff966
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.7.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_7_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_7_decode_layer_call.arg7.boundary_pack.reshard105'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0061 // fingerprint=c7c1586be4bff68dbfdf8ecd301e96344e03d6eaec455c28fb593c1665a6f3f0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.7.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.input_layernorm.weight')): // bf16<4>[512] name='layer_7_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_7_decode_layer_call.arg2.boundary_pack.reshard100'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_7_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0062 // fingerprint=62952d724e58ba48c3fe1c03e6b0ba7d665b10d8613998eb937cf4ea51cec9cc
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.7.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_7_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard101'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_7_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.7.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_7_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard102'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_7_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.7.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_7_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard103'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_7_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_7_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_7_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_7_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_7_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0063 // fingerprint=45c621ad1d3717f15035719f5b48a3940e400c025d92e499502407582d6f7bee
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.7.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_7_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard106'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_7_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0064 // fingerprint=0bf6e7da527d023cddb5b76805391b9c1d182e12ae062009a13d81e62b6941c6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.7.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_7_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_7_decode_layer_call.arg9.boundary_pack.reshard107'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_7_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0065 // fingerprint=211bbb70edc688d1614ad0c3a91dea5a25d20f514a2aedf2ffde67a9324c3099
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.7.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.7.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_7_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_7_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_7_decode_layer_call.gu_weight_rowmajor.reshard108'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_7_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0066 // fingerprint=b8d7518f6e5c2fc3c55c89d03e99b4e72977b27001df9aa7b71d926aa731a707
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.7.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.7.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_7_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard109'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_7_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0067 // fingerprint=13a19944fb3580af8fcf99d562e7c8fa53a757a14f37076059748b4c459558af
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.8.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_8_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_8_decode_layer_call.arg6.boundary_pack.reshard114'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0068 // fingerprint=f0d9e27a0d6bf3e30c254033c42b729420f045e7a168379bbc3771d6124c0c08
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.8.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_8_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_8_decode_layer_call.arg7.boundary_pack.reshard115'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0069 // fingerprint=612c3c1267d172ef96e75fa9cb838b831ca0831d6eb79754acf4f43b36db7382
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.8.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.input_layernorm.weight')): // bf16<4>[512] name='layer_8_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_8_decode_layer_call.arg2.boundary_pack.reshard110'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_8_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0070 // fingerprint=800fdc40fc68bdc4e12d8dee6a7fce78b9a7e92eec6457a2c39cfac28c07dc95
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.8.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_8_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard111'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_8_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.8.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_8_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard112'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_8_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.8.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_8_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard113'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_8_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_8_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_8_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_8_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_8_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0071 // fingerprint=f9c53704403a4d5d43c857a02c40299ec1be7b5aceac9fb15b5476500d808885
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.8.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_8_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard116'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_8_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0072 // fingerprint=1205a41c7459347eac11e5f032dd32f16567ad5724a864d22077b2438eeabd74
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.8.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_8_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_8_decode_layer_call.arg9.boundary_pack.reshard117'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_8_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0073 // fingerprint=bcc6fac8142c86b47a9301b29b10ce4864d3b2fb6eabbf77e439bc4a98948199
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.8.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.8.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_8_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_8_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_8_decode_layer_call.gu_weight_rowmajor.reshard118'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_8_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0074 // fingerprint=28f642fef5aed79eda588f0351d204ae148d925512b9d63f242e47691557a22a
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.8.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.8.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_8_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard119'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_8_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0075 // fingerprint=1a1f7e95f002d7ce643332ca94d62aa2cce0c55f9196580cf5846138a2e38aa3
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.9.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_9_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_9_decode_layer_call.arg6.boundary_pack.reshard124'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0076 // fingerprint=0e22fa9da070188c2b61b521ae420074cf882c553ba32999946ae02f36ab618a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.9.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_9_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_9_decode_layer_call.arg7.boundary_pack.reshard125'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0077 // fingerprint=7625b2b3ee32dc2e42a8bde773035ecc1ee20efa9638e9e3775c95f8dfcd8c4c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.9.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.input_layernorm.weight')): // bf16<4>[512] name='layer_9_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_9_decode_layer_call.arg2.boundary_pack.reshard120'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_9_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0078 // fingerprint=0fdda52b90944254d0cd0ccd43c431da5b8c169b263ba3633e23bfd1c032b017
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.9.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_9_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard121'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_9_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.9.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_9_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard122'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_9_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.9.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_9_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard123'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_9_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_9_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_9_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_9_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_9_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0079 // fingerprint=62337ab60d6944ee9f33aadcb15beb8490b14e0b44b5f80bc7e31bcbea942e88
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.9.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_9_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard126'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_9_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0080 // fingerprint=808f11813d9c56322a1f12a2279e50642ddf767a9279277602d276b0e877f696
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.9.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_9_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_9_decode_layer_call.arg9.boundary_pack.reshard127'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_9_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0081 // fingerprint=9b94aed24f1bed5ab88863455aed7ce212cdd9093662c7edc3d6c265685201d3
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.9.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.9.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_9_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_9_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_9_decode_layer_call.gu_weight_rowmajor.reshard128'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_9_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0082 // fingerprint=113d152c55d23066d09d23eba76805674070aa88b8a8b8539b2bb3d0482f322b
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.9.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.9.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_9_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard129'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_9_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0083 // fingerprint=bff0b22d2dcb73b95dc10eeb5842a559d90a687b700071095a1fc10e07884486
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.10.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_10_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_10_decode_layer_call.arg6.boundary_pack.reshard134'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0084 // fingerprint=dbcaf65c4e2a7945387401f69d7ba0dc42670218ef1ead392c6bd8fc08f072db
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.10.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_10_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_10_decode_layer_call.arg7.boundary_pack.reshard135'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0085 // fingerprint=07604d37cd6d3ec8b69d811955ed54c9747db72f5095e576cda758f67f69b59d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.10.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.input_layernorm.weight')): // bf16<4>[512] name='layer_10_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_10_decode_layer_call.arg2.boundary_pack.reshard130'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_10_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0086 // fingerprint=ce7dd1e8912d4321ac49e2bc75e06dfe20b8541acdb28cc643ad0ce9d74c9a39
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.10.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_10_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard131'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_10_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.10.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_10_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard132'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_10_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.10.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_10_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard133'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_10_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_10_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_10_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_10_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_10_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0087 // fingerprint=003db388b71cecab77ec6485f6c7c72b2e70f84a23f0dc5c69fc80d0deaee734
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.10.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_10_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard136'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_10_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0088 // fingerprint=9af03443e3bbba12bec72780a5c12933dc2fa12fd005a04b5a3c49831b841799
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.10.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_10_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_10_decode_layer_call.arg9.boundary_pack.reshard137'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_10_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0089 // fingerprint=310171febc938d0934e582202ae13f3f43314368a4444adf61128c848dc231b2
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.10.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.10.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_10_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_10_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_10_decode_layer_call.gu_weight_rowmajor.reshard138'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_10_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0090 // fingerprint=88c5192188ed0b5e3803b03b7539d9596fdcd0aced49e34c5b9b2ff5437baa3a
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.10.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.10.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_10_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard139'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_10_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0091 // fingerprint=c2e8f2ece4dc217ccce1457298f33ac99b3b73212bfeb02cdf3d423c0e5d088c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.11.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_11_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_11_decode_layer_call.arg6.boundary_pack.reshard144'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0092 // fingerprint=a72e1762d476eae74d066c8718228f02ce9b9a7c50b6d39dd78b14e856869d97
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.11.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_11_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_11_decode_layer_call.arg7.boundary_pack.reshard145'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0093 // fingerprint=8e0b38e72bc91130285438350d10a8b6a1a46cc432967ad707c89b18a9862670
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.11.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.input_layernorm.weight')): // bf16<4>[512] name='layer_11_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_11_decode_layer_call.arg2.boundary_pack.reshard140'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_11_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0094 // fingerprint=f34723eaefa25222ce8cf92c8c2f06855281acdffa14303bed1ca8be812b9790
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.11.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_11_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard141'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_11_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.11.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_11_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard142'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_11_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.11.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_11_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard143'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_11_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_11_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_11_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_11_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_11_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0095 // fingerprint=56178126c426f59d3b29a426acdbe8daf376f8c8c55a49d9915d3f5d42f16842
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.11.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_11_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard146'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_11_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0096 // fingerprint=c283854f6a6c7a88792efc290d46ee10cd01093be23d03e35b09f72ccb83ab6b
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.11.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_11_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_11_decode_layer_call.arg9.boundary_pack.reshard147'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_11_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0097 // fingerprint=eb94fcce7d254901d88722a637aac293f357ed9fe5e7e775a299550438e9cf2c
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.11.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.11.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_11_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_11_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_11_decode_layer_call.gu_weight_rowmajor.reshard148'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_11_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0098 // fingerprint=ac55b42c6a7a2964ba8762a1f9240c2915da3ac9271c2d9e7126abc57d2207c3
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.11.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.11.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_11_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard149'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_11_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0099 // fingerprint=2fd75ecf8f22bad1585fec4ed5d2e191366cc8b15b9b957eb6c2cd77f90eba60
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.12.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_12_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_12_decode_layer_call.arg6.boundary_pack.reshard154'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0100 // fingerprint=e0440087fd7605e7c474b3c6c7c4d3eed98722d13ec393745a92a6e3ad425bb7
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.12.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_12_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_12_decode_layer_call.arg7.boundary_pack.reshard155'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0101 // fingerprint=e79ce7aa023b5e97a0727cefe0b1715a4db32fcbfffdd0e1343f2738eeaeaaef
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.12.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.input_layernorm.weight')): // bf16<4>[512] name='layer_12_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_12_decode_layer_call.arg2.boundary_pack.reshard150'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_12_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0102 // fingerprint=7e7ab142288db8418833bd7320a6d227fe74be8b9b2ea53ccfecbd701df88e60
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.12.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_12_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard151'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_12_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.12.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_12_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard152'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_12_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.12.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_12_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard153'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_12_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_12_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_12_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_12_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_12_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0103 // fingerprint=71080737c77efb17f64fe750960da0c195ed1198295c5a33e50e9c911d9d0f23
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.12.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_12_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard156'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_12_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0104 // fingerprint=beb7e70f1c0248b32661b708e6d5dc08804bb344f8219e1ed8320547cf9a1841
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.12.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_12_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_12_decode_layer_call.arg9.boundary_pack.reshard157'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_12_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0105 // fingerprint=50d7ab8ecff0308c15d56453a94f0ab6b0f2f5ebdd76d3d4e335adea080cae2a
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.12.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.12.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_12_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_12_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_12_decode_layer_call.gu_weight_rowmajor.reshard158'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_12_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0106 // fingerprint=acca4fb888df2ec91c12ffeec3b45a11ba7443c36be502e6ebfac3c736c641f2
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.12.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.12.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_12_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard159'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_12_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0107 // fingerprint=849e5f3c7919503190a888b6f4d492614ee8bc14fa5145e2156977c6d48fbec4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.13.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_13_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_13_decode_layer_call.arg6.boundary_pack.reshard164'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0108 // fingerprint=e4ad538a1551a40266e0260b57b2c8bfb6a0d6230cc3a9229e71a36c7a1bbc16
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.13.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_13_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_13_decode_layer_call.arg7.boundary_pack.reshard165'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0109 // fingerprint=1ac21ec8eefd20d9eaa0e7b1e7f0b52d647a69afc26e8e4fb5895c0f492fe81a
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.13.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.input_layernorm.weight')): // bf16<4>[512] name='layer_13_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_13_decode_layer_call.arg2.boundary_pack.reshard160'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_13_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0110 // fingerprint=a4cb6ab8b1bb32cc8bfb1a96d7b56f5a224a0184287ede49493a56280c979b73
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.13.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_13_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard161'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_13_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.13.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_13_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard162'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_13_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.13.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_13_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard163'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_13_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_13_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_13_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_13_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_13_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0111 // fingerprint=98a1f475fa683549e9816833fd265db536dabac54867b04197f24f8f0596ec08
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.13.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_13_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard166'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_13_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0112 // fingerprint=9cdeab59bd6e4757022034d22371ad3f92f2c2908f188c17d28dd332dfe087c2
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.13.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_13_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_13_decode_layer_call.arg9.boundary_pack.reshard167'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_13_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0113 // fingerprint=e972d9e7a208eb3eb66100f5a54f46bea79b9abce82770d704f9307d9242a89f
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.13.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.13.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_13_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_13_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_13_decode_layer_call.gu_weight_rowmajor.reshard168'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_13_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0114 // fingerprint=e66153dd09f1234c789e83684efac5d0229ae4b4a2b3110b42c7af3b72733a14
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.13.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.13.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_13_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard169'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_13_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0115 // fingerprint=8a0d12d42666b4a8e01400ff9c5f22415c156da81d71e7e9ba731815cd42001f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.14.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_14_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_14_decode_layer_call.arg6.boundary_pack.reshard174'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0116 // fingerprint=bbded5b8d950d696b09ef4dc1d0d7d237edabdd8dde9c0b1d3c3d73053faa55c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.14.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_14_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_14_decode_layer_call.arg7.boundary_pack.reshard175'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0117 // fingerprint=6824d1025a8bed8544cb623bfca24ebe85acdea978208688029f1e21f307c0a6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.14.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.input_layernorm.weight')): // bf16<4>[512] name='layer_14_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_14_decode_layer_call.arg2.boundary_pack.reshard170'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_14_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0118 // fingerprint=9471b617ffc52965beaa49a9b015e8a02c295274123200b11f4ac0ba5292dbcf
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.14.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_14_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard171'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_14_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.14.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_14_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard172'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_14_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.14.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_14_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard173'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_14_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_14_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_14_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_14_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_14_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0119 // fingerprint=e98edf1f69223cd47d0740d051aae2f215757a7d6cfb1734faf1b1adcc4c60b6
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.14.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_14_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard176'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_14_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0120 // fingerprint=92ce87e0d1981755b29125143f2d63573902063d2b0384b71aa6be8734c32a7c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.14.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_14_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_14_decode_layer_call.arg9.boundary_pack.reshard177'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_14_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0121 // fingerprint=f56e14c6cad798720ec2de2d7d5bde3c933d73ad67a42da0343b1b2712af6d92
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.14.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.14.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_14_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_14_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_14_decode_layer_call.gu_weight_rowmajor.reshard178'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_14_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0122 // fingerprint=9cbd401d9d334216f072c3606a540a908e963335a9567d880d77eaf365910c98
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.14.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.14.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_14_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard179'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_14_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0123 // fingerprint=36e9b2c2ca3a36f86875548410081d48c90016fb372efbf9c94fcea2218d435d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.15.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_15_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_15_decode_layer_call.arg6.boundary_pack.reshard184'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0124 // fingerprint=7dc2a6b5b13b077ab29b49d2cda9e87d5615fa40cd3a175b224ba430a8eba870
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.15.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_15_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_15_decode_layer_call.arg7.boundary_pack.reshard185'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0125 // fingerprint=e8dab58e1d9b47be39fc586be12f3c492e40694f8c3ee08181d1b86959bf1a59
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.15.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.input_layernorm.weight')): // bf16<4>[512] name='layer_15_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_15_decode_layer_call.arg2.boundary_pack.reshard180'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_15_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0126 // fingerprint=6dd4ba6d66d71678a1df855d4d9433ad9943e2494607fadce88a0a68aec481bc
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.15.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_15_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard181'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_15_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.15.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_15_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard182'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_15_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.15.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_15_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard183'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_15_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_15_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_15_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_15_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_15_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0127 // fingerprint=5e6fe3764b4f9a118ab43ca0de7a7547cd73c6d62c36f582d87489c13b0796dc
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.15.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_15_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard186'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_15_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0128 // fingerprint=334a1c1854f0d9f486f6fc8a873a4094e06c4139b1d7ad6edececf97555a473d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.15.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_15_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_15_decode_layer_call.arg9.boundary_pack.reshard187'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_15_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0129 // fingerprint=2e44cbca09d7f7e48856c50f1ff6a18d0455425267d7c4bfc02cdaf4f98c2fc8
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.15.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.15.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_15_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_15_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_15_decode_layer_call.gu_weight_rowmajor.reshard188'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_15_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0130 // fingerprint=9c6c3810e8612f30c08f750909797aa49cd4bc3786b6de6eab830ddd56a3a8ce
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.15.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.15.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_15_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard189'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_15_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0131 // fingerprint=d167dd102f4cfc494f39aa328e002431b1f299476d813b37e7ec0fe75189e103
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.16.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_16_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_16_decode_layer_call.arg6.boundary_pack.reshard194'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0132 // fingerprint=527ade5970b4a4270b201580c026f1085eed45f6ab633c5535a1e5e47ec601eb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.16.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_16_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_16_decode_layer_call.arg7.boundary_pack.reshard195'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0133 // fingerprint=e44f12a2ba10565374e8280aa7ae68c5b468dc9c5e4db2ebe676401d3b2711b4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.16.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.input_layernorm.weight')): // bf16<4>[512] name='layer_16_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_16_decode_layer_call.arg2.boundary_pack.reshard190'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_16_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0134 // fingerprint=163b45e8a4b8a9ad1de5dbf7952b902912e90933415da070c0f54920dc6941fa
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.16.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_16_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard191'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_16_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.16.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_16_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard192'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_16_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.16.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_16_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard193'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_16_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_16_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_16_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_16_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_16_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0135 // fingerprint=ef04dc48fd16b23a08116aaa12e24bf7b2d6f52abdac5e8c7698263851eefffd
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.16.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_16_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard196'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_16_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0136 // fingerprint=bc626ffe4fbb6c87cf91c13e5ff32b26334b1d1453aec69b7aa4f73e5d3cd129
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.16.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_16_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_16_decode_layer_call.arg9.boundary_pack.reshard197'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_16_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0137 // fingerprint=9c210631c65b7293ed19b23e5f9cde10023c6acbcd8d9b66eee2318fa6691739
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.16.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.16.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_16_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_16_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_16_decode_layer_call.gu_weight_rowmajor.reshard198'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_16_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0138 // fingerprint=9b6a2d57369478f364334ed4a7b07d4cca7bb1929221d14749f4287967de676c
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.16.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.16.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_16_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard199'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_16_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0139 // fingerprint=783863b6ea181b4c2428811ed774a448dfb3098cfb20132dea75a245121d5db6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.17.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_17_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_17_decode_layer_call.arg6.boundary_pack.reshard204'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0140 // fingerprint=983dd7ca682bc7f0980892a66a85e49d69fc173fb0126ddb9c8e2a1e61e18580
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.17.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_17_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_17_decode_layer_call.arg7.boundary_pack.reshard205'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0141 // fingerprint=659c4eac1f9976db6f0c11095ee837e18c3d215a7552bda63b8338e0b527b99d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.17.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.input_layernorm.weight')): // bf16<4>[512] name='layer_17_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_17_decode_layer_call.arg2.boundary_pack.reshard200'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_17_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0142 // fingerprint=22aab3e75f12abf6ff9593eb5bb961101169e7fb678c3096f1f81b187810b5e9
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.17.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_17_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard201'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_17_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.17.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_17_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard202'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_17_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.17.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_17_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard203'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_17_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_17_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_17_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_17_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_17_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0143 // fingerprint=16120e0fa7759008eeedb5facb31e96356da5647ac274f6c83ec077857dc996f
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.17.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_17_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard206'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_17_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0144 // fingerprint=e9f94953b3ad033e2ce7f4a2c9fe809cdbc82843df42ec93ea6acd10e0c536ad
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.17.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_17_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_17_decode_layer_call.arg9.boundary_pack.reshard207'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_17_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0145 // fingerprint=d3baf632e19248f203e222059e3a7c63ac307db00c5bfe552dce10a031c8c3ed
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.17.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.17.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_17_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_17_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_17_decode_layer_call.gu_weight_rowmajor.reshard208'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_17_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0146 // fingerprint=b952838b29110656721f39dd93ffb13394611ddac74cad1978044269e4ca7576
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.17.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.17.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_17_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard209'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_17_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0147 // fingerprint=d88e47ee9af433d8d2359ffc7d7e74d62f84efae37e4b5591a5a958af3e7e5ea
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.18.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_18_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_18_decode_layer_call.arg6.boundary_pack.reshard214'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0148 // fingerprint=ed48bdd6b9b74430b03699a9cb746bcf0c5fd12f8ab89b0bc239eb62bf0339fd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.18.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_18_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_18_decode_layer_call.arg7.boundary_pack.reshard215'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0149 // fingerprint=9272cc0106457d667b56c50b6f134b37d299ee4f92a127c0484e0dfbd3c7febb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.18.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.input_layernorm.weight')): // bf16<4>[512] name='layer_18_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_18_decode_layer_call.arg2.boundary_pack.reshard210'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_18_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0150 // fingerprint=52d9292ec0b29a195e4f3b94b4a7c9c97f259ea4a9aafb53a0fe3fe4ebde097b
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.18.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_18_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard211'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_18_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.18.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_18_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard212'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_18_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.18.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_18_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard213'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_18_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_18_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_18_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_18_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_18_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0151 // fingerprint=d4813a1423bff564169b1fc22943bbd32fa9c11676e2bcd3e0f921b6a9c01644
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.18.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_18_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard216'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_18_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0152 // fingerprint=c625ae5c105236fc715a375c0f8024107e775834e23a4e2934f67e85ec040bda
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.18.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_18_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_18_decode_layer_call.arg9.boundary_pack.reshard217'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_18_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0153 // fingerprint=16e791aa0b072d538b11537b579c46132e323888afc7132b00706acdd9cb1248
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.18.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.18.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_18_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_18_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_18_decode_layer_call.gu_weight_rowmajor.reshard218'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_18_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0154 // fingerprint=58747f5597ac6b48d1d1f3736396b7530ca08add834b3e9a7db47f4a318ccf03
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.18.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.18.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_18_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard219'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_18_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0155 // fingerprint=26439b80677bbbaceeec831a46ec5f63b0fa1b82e6a23216cd4afa80945d1fb4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.19.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_19_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_19_decode_layer_call.arg6.boundary_pack.reshard224'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0156 // fingerprint=26b4f5793ad8343f5a065949924ccb9799f4321f2c59f9ed4455775529e45a5f
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.19.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_19_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_19_decode_layer_call.arg7.boundary_pack.reshard225'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0157 // fingerprint=125e01748666437caa5f08f9a5259575bc6575a61f756f3a6f9b6572691b42c1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.19.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.input_layernorm.weight')): // bf16<4>[512] name='layer_19_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_19_decode_layer_call.arg2.boundary_pack.reshard220'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_19_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0158 // fingerprint=39ff89335887629826a4090c69a9c317dc42bad8b5ed26af97179ce736a0d015
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.19.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_19_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard221'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_19_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.19.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_19_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard222'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_19_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.19.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_19_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard223'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_19_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_19_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_19_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_19_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_19_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0159 // fingerprint=40700cc217f3e89e1626ffb3d260ed8c6d92e4c0d38ce720b9273039e624adee
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.19.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_19_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard226'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_19_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0160 // fingerprint=cc151c7b01dbd27e3efe1866f07fa4d6d65bb6057ef9c117542e206a91e4eda7
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.19.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_19_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_19_decode_layer_call.arg9.boundary_pack.reshard227'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_19_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0161 // fingerprint=d73b7e92aaee82ff572d730f854f64154cb9e3c50b64746f56cef02dc873eed2
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.19.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.19.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_19_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_19_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_19_decode_layer_call.gu_weight_rowmajor.reshard228'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_19_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0162 // fingerprint=81a905b969b0baafe212363d243564a2d74089dda42dbded3e0e3c7cd48649d2
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.19.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.19.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_19_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard229'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_19_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0163 // fingerprint=cc97279c91c2694a996e3ebdb35cb2d727a8edd0cd24acf33e9a5b86f5d985dd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.20.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_20_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_20_decode_layer_call.arg6.boundary_pack.reshard234'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0164 // fingerprint=b723d4700b27d88e5269f39190a1cd6c3de6acd288ea693bea2d54dca86648f9
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.20.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_20_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_20_decode_layer_call.arg7.boundary_pack.reshard235'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0165 // fingerprint=672c1d95af6549545b7803cfcdc1a8a1e2c1ec2cf157bcd9a4db0f86f503aeb6
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.20.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.input_layernorm.weight')): // bf16<4>[512] name='layer_20_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_20_decode_layer_call.arg2.boundary_pack.reshard230'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_20_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0166 // fingerprint=0f41398afeec66506dd45684f8d423f76a4ba85ec33fc7ad9b6b11e63807fb4d
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.20.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_20_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard231'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_20_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.20.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_20_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard232'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_20_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.20.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_20_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard233'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_20_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_20_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_20_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_20_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_20_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0167 // fingerprint=05258b615d0f717a128e21c7e4b9f126a7b9b8517f009c47fdda3c040328b2e1
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.20.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_20_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard236'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_20_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0168 // fingerprint=3aa8ff8fea4b968ff0f5efb84df5d47b123aa584097b8f0ffd65919913f737ef
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.20.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_20_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_20_decode_layer_call.arg9.boundary_pack.reshard237'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_20_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0169 // fingerprint=086514cd71705b859ef2de9e42855c14ad672eb8af815e64b30174304a902087
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.20.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.20.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_20_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_20_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_20_decode_layer_call.gu_weight_rowmajor.reshard238'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_20_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0170 // fingerprint=4cecea781dd792c7312c8f39d7677e1be5a7ff5e10f367ec4e75f6f6fe32d493
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.20.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.20.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_20_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard239'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_20_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0171 // fingerprint=846645d7789dfb1c8b870662e4c7a86a2e84efc9872851b6db0f73f5e778124b
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.21.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_21_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_21_decode_layer_call.arg6.boundary_pack.reshard244'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0172 // fingerprint=59523ee29adfa19219a9b09d16feab43b4afbc75b0d433e06d5289cf780134d4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.21.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_21_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_21_decode_layer_call.arg7.boundary_pack.reshard245'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0173 // fingerprint=5b1e8e4b8550d0e31b82f1894d4ee599fccb52fac2e38d380cdf92e06487aa6c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.21.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.input_layernorm.weight')): // bf16<4>[512] name='layer_21_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_21_decode_layer_call.arg2.boundary_pack.reshard240'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_21_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0174 // fingerprint=541a440f29ce1f4db3f39c6e1aec3f8e0a0a5dd4f673c0410bb657363f5144af
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.21.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_21_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard241'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_21_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.21.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_21_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard242'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_21_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.21.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_21_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard243'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_21_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_21_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_21_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_21_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_21_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0175 // fingerprint=9f2be527225820fc86716de97393168c43ae24060c674cbde4374615f4508565
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.21.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_21_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard246'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_21_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0176 // fingerprint=cb7af72d54ef38d4d9dc3051e223998b37bf2f55fdfc897c9bbfddf343212e03
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.21.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_21_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_21_decode_layer_call.arg9.boundary_pack.reshard247'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_21_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0177 // fingerprint=6031976abf43b06d414ea81b6fd0757d8e51f92aa06958b8e21f4558d299039f
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.21.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.21.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_21_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_21_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_21_decode_layer_call.gu_weight_rowmajor.reshard248'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_21_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0178 // fingerprint=8bc72c8fe8d6e584a25fe64c3ddbe09375979a81814e4fb2488912b0a1dd4569
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.21.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.21.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_21_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard249'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_21_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0179 // fingerprint=7f8bd1facf8b68a6706c6917bcfcd31b3a5985d76cae404564b5ac017c722c14
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.22.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_22_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_22_decode_layer_call.arg6.boundary_pack.reshard254'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0180 // fingerprint=131c24a4ca3847c33948c505d8aa2c3b7680d073e4e08c27c75a832bbff3a38d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.22.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_22_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_22_decode_layer_call.arg7.boundary_pack.reshard255'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0181 // fingerprint=e6850103519dbdfa901e4b3123811fbb11590d9bea7a61801785d0e4298b4c19
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.22.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.input_layernorm.weight')): // bf16<4>[512] name='layer_22_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_22_decode_layer_call.arg2.boundary_pack.reshard250'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_22_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0182 // fingerprint=95b26be00037744ad0d66216b970e099dbd4b945dd55c97c1b152e0878bc2da5
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.22.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_22_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard251'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_22_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.22.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_22_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard252'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_22_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.22.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_22_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard253'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_22_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_22_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_22_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_22_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_22_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0183 // fingerprint=5119960b20aefcb89b910ccfb6219a157388a827b27a4a5a6595e3d70c5f1576
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.22.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_22_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard256'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_22_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0184 // fingerprint=aba514c0f8bc9fb5d6a3f7aa357ec4d9c76a9e47535a793d61a703fcdefc1fe1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.22.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_22_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_22_decode_layer_call.arg9.boundary_pack.reshard257'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_22_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0185 // fingerprint=34e13235cbbd852049a11a0675ad031a5037b52df4d32e6fc64f41c868b4d48e
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.22.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.22.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_22_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_22_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_22_decode_layer_call.gu_weight_rowmajor.reshard258'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_22_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0186 // fingerprint=6d590c08d8afacc59ec92ae7dc98f66004548eeda7035e42901df7ffa35c5bc4
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.22.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.22.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_22_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard259'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_22_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0187 // fingerprint=c6b1925b7b4a07fec1c025d590f5e9b74b1e7386730716d75a4518eeb075a302
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.23.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_23_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_23_decode_layer_call.arg6.boundary_pack.reshard264'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0188 // fingerprint=36a3d38c27943cec1e50edf0e26112e8e2ebc7312149ee8c5916dd823556021c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.23.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_23_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_23_decode_layer_call.arg7.boundary_pack.reshard265'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0189 // fingerprint=a5f81ae71dda72e151b72cf9197e885b3b8ccc28b5774634ed3bfe61583005d4
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.23.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.input_layernorm.weight')): // bf16<4>[512] name='layer_23_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_23_decode_layer_call.arg2.boundary_pack.reshard260'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_23_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0190 // fingerprint=a683ba4b9a9c5a7638c548a4f7e5908f21bf2f5c868d3258b67468f924c1a33e
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.23.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_23_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard261'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_23_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.23.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_23_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard262'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_23_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.23.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_23_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard263'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_23_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_23_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_23_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_23_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_23_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0191 // fingerprint=f513fa38e7b913535372b4216f51f7a6be8cef1f1c3ab059f0d53907e5716728
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.23.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_23_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard266'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_23_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0192 // fingerprint=79d714e2fdd6780ed537c83b73238f6ad3ed808046e49c89c118227386acdd5d
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.23.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_23_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_23_decode_layer_call.arg9.boundary_pack.reshard267'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_23_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0193 // fingerprint=c2e1973b2632d928cc6b674bfd452452bc375f723e39c8028eccdd1fc3a099cf
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.23.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.23.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_23_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_23_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_23_decode_layer_call.gu_weight_rowmajor.reshard268'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_23_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0194 // fingerprint=0cdf24d4f6fa37e07e89e0e224e2e92f095f641383c81cd4a2c64c3357bcab73
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.23.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.23.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_23_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard269'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_23_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0195 // fingerprint=01741f03d44130c066b9b26437656127bc2671881a1f68508893d3c309fc060c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.24.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_24_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_24_decode_layer_call.arg6.boundary_pack.reshard274'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0196 // fingerprint=cb7eea23432de78782b9cb7cb0d9a56ca8c9e12fc6f97eebaa7b1754c556bce9
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.24.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_24_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_24_decode_layer_call.arg7.boundary_pack.reshard275'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0197 // fingerprint=d9c899812e20c7772fdb8bda1d91e21a60b2426014b62c6bd10cbd2cfd504834
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.24.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.input_layernorm.weight')): // bf16<4>[512] name='layer_24_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_24_decode_layer_call.arg2.boundary_pack.reshard270'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_24_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0198 // fingerprint=d98ae6ff3d97f0ca3588fc0ac7023c3f354b4dee59eaf7e971f9dbd03c64c27c
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.24.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_24_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard271'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_24_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.24.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_24_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard272'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_24_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.24.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_24_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard273'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_24_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_24_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_24_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_24_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_24_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0199 // fingerprint=55ee05820986f656e828231bcc425e72b130e6defb55512f237e57aabb8db675
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.24.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_24_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard276'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_24_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0200 // fingerprint=15d9a6a05d86f3fcb11b6cb3b081972fa55a112c09c3e1efc65bbfecaa7ebfe0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.24.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_24_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_24_decode_layer_call.arg9.boundary_pack.reshard277'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_24_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0201 // fingerprint=ec45126eca7e921aedaafe78d73d62c18888395befc0d2232597f6613291b638
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.24.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.24.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_24_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_24_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_24_decode_layer_call.gu_weight_rowmajor.reshard278'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_24_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0202 // fingerprint=2a5462339f30b8de02ec01f1d641358015f16175e6e6965f9d6718e71c2937d4
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.24.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.24.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_24_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard279'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_24_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0203 // fingerprint=72544951d76a40a4d5ab5180eb34565cdeace89817e7e6a457cb06af33cd3359
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.25.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_25_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_25_decode_layer_call.arg6.boundary_pack.reshard284'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0204 // fingerprint=c4808b1a2bb0a01f77c64f2438dce72115ad3091f41ce876286536e756db728c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.25.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_25_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_25_decode_layer_call.arg7.boundary_pack.reshard285'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0205 // fingerprint=a146661de56ef9f23c217b05634f7c3b0fe9458481bb1596cc5d9ae8b46bd44e
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.25.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.input_layernorm.weight')): // bf16<4>[512] name='layer_25_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_25_decode_layer_call.arg2.boundary_pack.reshard280'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_25_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0206 // fingerprint=5f3e5864a8f78bc2edc22857a668301e690675f332c81448e9dc01953d093a27
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.25.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_25_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard281'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_25_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.25.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_25_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard282'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_25_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.25.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_25_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard283'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_25_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_25_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_25_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_25_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_25_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0207 // fingerprint=4953a59d14ddbaada8dc413b94a9206df981e93ee4692752ef23308d29479ab9
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.25.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_25_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard286'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_25_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0208 // fingerprint=e94476c16a559660c91130a9572d9c2a10af417930cae5afd431d5164ff190cd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.25.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_25_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_25_decode_layer_call.arg9.boundary_pack.reshard287'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_25_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0209 // fingerprint=281fe549901d005a6a04db3f5bd5ef7f42d7f7c7bf8542a3a65da859df19a096
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.25.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.25.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_25_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_25_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_25_decode_layer_call.gu_weight_rowmajor.reshard288'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_25_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0210 // fingerprint=ea5e82d498b7ff69890c9bc83995731a9abd7db654cc8a9686b54a360288f0a0
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.25.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.25.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_25_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard289'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_25_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0211 // fingerprint=a654719eeb0879ef06d4b2afc33dc2f08f6d40e06b256804a799e1dd3ace85a0
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.26.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_26_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_26_decode_layer_call.arg6.boundary_pack.reshard294'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0212 // fingerprint=5258f7b9ac45bc576b6aa7971cb265a915aff95b5f619bc4a73a4b5be173b1cb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.26.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_26_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_26_decode_layer_call.arg7.boundary_pack.reshard295'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0213 // fingerprint=25e1b00911b7436ff77bd2248d6b5a0cc060924c3eb9e6f904d23844656db209
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.26.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.input_layernorm.weight')): // bf16<4>[512] name='layer_26_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_26_decode_layer_call.arg2.boundary_pack.reshard290'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_26_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0214 // fingerprint=6ec4459e56cc0bf4477e6d2eb6ab5b9250c3f04ecd74526565dc946282393b46
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.26.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_26_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard291'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_26_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.26.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_26_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard292'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_26_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.26.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_26_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard293'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_26_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_26_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_26_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_26_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_26_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0215 // fingerprint=a4415fa2e016515e7f5a1b662c5b48c4277a1ab0fe7436739408f64c8d9bd3a1
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.26.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_26_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard296'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_26_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0216 // fingerprint=efc11e28cc4112f4d3828129766b52eb4bc103d8b22ebc8fef33f7b8af1fe0dd
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.26.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_26_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_26_decode_layer_call.arg9.boundary_pack.reshard297'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_26_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0217 // fingerprint=fb0e3566373cd132e042b723f4f5bf8eca38df88facb1648250fc81a884e8f22
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.26.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.26.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_26_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_26_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_26_decode_layer_call.gu_weight_rowmajor.reshard298'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_26_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0218 // fingerprint=01485f65315a0a58125f7b1601ffc18960366355cbf36ef0a58225055cca1fdd
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.26.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.26.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_26_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard299'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_26_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0219 // fingerprint=f4e9f3abf74349cd569d121beb7ce7de2ed483a79017d45bb5d777da7af43ab1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.27.self_attn.q_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.q_norm.weight')): // bf16<8>[16] name='layer_27_decode_layer_call.arg6.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_27_decode_layer_call.arg6.boundary_pack.reshard304'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0220 // fingerprint=dd7939992141421cd05c28b40b2652f568f9421ffdb202a860add4de855d7e0b
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [8], T.WeightRef(bf16[128], 'layers.27.self_attn.k_norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.k_norm.weight')): // bf16<8>[16] name='layer_27_decode_layer_call.arg7.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<8>[16], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<8>[16], (B), [16], ) name='layer_27_decode_layer_call.arg7.boundary_pack.reshard305'
      yield (%1)
    }
    T.ConstantRecipe constant_recipe_0221 // fingerprint=31df1e26a0c7a6a3f94760b3f3dac5d2602a5dbdcad6b958ef71e4e63bc6cac1
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.27.input_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.input_layernorm.weight')): // bf16<4>[512] name='layer_27_decode_layer_call.arg2.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_27_decode_layer_call.arg2.boundary_pack.reshard300'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(64)@[0])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ) name='layer_27_decode_layer_call.arg2.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0222 // fingerprint=33b9d50bbbd92b355a9119bd47f854c8eff07b1dd2212c6f091a9afd0a8dae39
    {
      %0  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.27.self_attn.q_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.q_proj.weight')): // bf16<8>[2048,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %1  = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %2  = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %3  = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector.fold_pack_transpose.pack'
      %4  = Tensors.Permute(Axes: [1, 0], %3): // bf16<8,2,8>[256,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.n_vector'
      %5  = Tensors.Permute(Axes: [1, 0], %4): // bf16<8,2,8>[128,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major'
      %6  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %5): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_27_decode_layer_call.decode_layer_qkv_projection.q_weight_pack.k_major.reshard301'
      %7  = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %6): // Dist(bf16<8,2,8>[128,256], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,16@x], ) name='layer_27_decode_layer_call.arg3.distributed_boundary_reshard'
      %8  = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.27.self_attn.k_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.k_proj.weight')): // bf16<8>[1024,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %9  = Tensors.Pack(Axes: [1], Lanes: [2], %8): // bf16<2,8>[1024,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %10 = Tensors.Pack(Axes: [0], Lanes: [8], %9): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %11 = Tensors.Permute(Axes: [1, 0], %10): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector.fold_pack_transpose.pack'
      %12 = Tensors.Permute(Axes: [1, 0], %11): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.n_vector'
      %13 = Tensors.Permute(Axes: [1, 0], %12): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major'
      %14 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %13): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_27_decode_layer_call.decode_layer_qkv_projection.k_weight_pack.k_major.reshard302'
      %15 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %14): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_27_decode_layer_call.arg4.distributed_boundary_reshard'
      %16 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[1024,2048], 'layers.27.self_attn.v_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.v_proj.weight')): // bf16<8>[1024,256] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %17 = Tensors.Pack(Axes: [1], Lanes: [2], %16): // bf16<2,8>[1024,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_pack.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %18 = Tensors.Pack(Axes: [0], Lanes: [8], %17): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack.fold_pack_transpose.pack'
      %19 = Tensors.Permute(Axes: [1, 0], %18): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector.fold_pack_transpose.pack'
      %20 = Tensors.Permute(Axes: [1, 0], %19): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.n_vector'
      %21 = Tensors.Permute(Axes: [1, 0], %20): // bf16<8,2,8>[128,128] name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major'
      %22 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (B,B), [y:8,x:16], Partial: ), %21): // Dist(bf16<8,2,8>[128,128], (B,B), [128,128], ) name='layer_27_decode_layer_call.decode_layer_qkv_projection.v_weight_pack.k_major.reshard303'
      %23 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [y:8,x:16], Partial: ), %22): // Dist(bf16<8,2,8>[128,128], (S(C(16)@[0]),S(BC(8)@[1])), [16@y,8@x], ) name='layer_27_decode_layer_call.arg5.distributed_boundary_reshard'
      %24 = Distributed.MaterializeLocalShards(%7): // bf16<8,2,8>[128,16,16] name='layer_27_decode_layer_call.qkv_fused.source_0'
      %25 = Distributed.MaterializeLocalShards(%15): // bf16<8,2,8>[128,16,8] name='layer_27_decode_layer_call.qkv_fused.source_1'
      %26 = Distributed.MaterializeLocalShards(%23): // bf16<8,2,8>[128,16,8] name='layer_27_decode_layer_call.qkv_fused.source_2'
      %27 = Tensors.Concat(Axis: 2, %24, %25, %26): // bf16<8,2,8>[128,16,32] name='layer_27_decode_layer_call.qkv_fused'
      yield (%27)
    }
    T.ConstantRecipe constant_recipe_0223 // fingerprint=7ebcef0466c90e6efd563c02484283dcc20b28c34c47a4a3bf3b0ed4e608f860
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,2048], 'layers.27.self_attn.o_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.self_attn.o_proj.weight')): // bf16<8>[2048,256] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,128] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,128] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[128,256] name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[128,256], (B,B), [128,256], ) name='layer_27_decode_layer_call.decode_layer_attention_output.vectorized.compute.rhs_k_major.physical.reshard306'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ) name='layer_27_decode_layer_call.arg8.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0224 // fingerprint=a99e58b02805ec3378b5534be6d06e0f0a2716625ac433d8ecb31b7db4237ffb
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'layers.27.post_attention_layernorm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.post_attention_layernorm.weight')): // bf16<4>[512] name='layer_27_decode_layer_call.arg9.boundary_pack'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (B), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (B), [512], ) name='layer_27_decode_layer_call.arg9.boundary_pack.reshard307'
      %2 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %1): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='layer_27_decode_layer_call.arg9.distributed_boundary_reshard'
      yield (%2)
    }
    T.ConstantRecipe constant_recipe_0225 // fingerprint=18570d7a91daf5745be5a4db5e66c96926c77003e7d1938ce4f8a07359a4cf4c
    {
      %0 = Tensors.Concat(Axis: 0, T.WeightRef(bf16[6144,2048], 'layers.27.mlp.gate_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.mlp.gate_proj.weight'), T.WeightRef(bf16[6144,2048], 'layers.27.mlp.up_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.mlp.up_proj.weight')): // bf16[12288,2048] name='layer_27_decode_layer_call.gu_weight_join'
      %1 = Tensors.Permute(Axes: [1, 0], %0): // bf16[2048,12288] name='layer_27_decode_layer_call.gu_weight_rowmajor'
      %2 = Distributed.ShardedView(NewType: Dist(bf16[2048,12288], (B,B), [y:8,x:16], Partial: ), %1): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_27_decode_layer_call.gu_weight_rowmajor.reshard308'
      %3 = Distributed.Boxing(NewType: Dist(bf16[2048,12288], (B,B), [y:48,x:2], Partial: ), %2): // Dist(bf16[2048,12288], (B,B), [2048,12288], ) name='layer_27_decode_layer_call.arg10.distributed_boundary_reshard'
      yield (%3)
    }
    T.ConstantRecipe constant_recipe_0226 // fingerprint=374182c3d80bd557f9f09c2bcdf1291b50994827d0b641784f465a56be31e9f7
    {
      %0 = Tensors.Pack(Axes: [1], Lanes: [8], T.WeightRef(bf16[2048,6144], 'layers.27.mlp.down_proj.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.layers.27.mlp.down_proj.weight')): // bf16<8>[2048,768] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_vector'
      %1 = Tensors.Pack(Axes: [1], Lanes: [2], %0): // bf16<2,8>[2048,384] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.k_pack'
      %2 = Tensors.Pack(Axes: [0], Lanes: [8], %1): // bf16<8,2,8>[256,384] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.n_vector'
      %3 = Tensors.Permute(Axes: [1, 0], %2): // bf16<8,2,8>[384,256] name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical'
      %4 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,B), [y:8,x:16], Partial: ), %3): // Dist(bf16<8,2,8>[384,256], (B,B), [384,256], ) name='layer_27_decode_layer_call.decode_layer_mlp_down.vectorized.compute.rhs_k_major.physical.reshard309'
      %5 = Distributed.ShardedView(NewType: Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [y:8,x:16], Partial: ), %4): // Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ) name='layer_27_decode_layer_call.arg11.distributed_boundary_reshard'
      yield (%5)
    }
    T.ConstantRecipe constant_recipe_0227 // fingerprint=aa591a81a5de287d0a3c96d795aa00689da07bf8bac76c682dbfa8ebbbc3b77c
    {
      %0 = Tensors.Pack(Axes: [0], Lanes: [4], T.WeightRef(bf16[2048], 'norm.weight', Source: 'model-00001-of-00002.safetensors', Key: 'model.norm.weight')): // bf16<4>[512] name='final_norm.vectorized.pack2'
      %1 = Distributed.ShardedView(NewType: Dist(bf16<4>[512], (S(C(4)@[0,1])), [y:8,x:16], Partial: ), %0): // Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ) name='final_norm.vectorized.pack2.reshard311'
      yield (%1)
    }
    %w0 = None() // none name='decode_layer_qkv_none'
  }
  // compute
  %0   = Distributed.ShardedView(NewType: Dist(i32[1], (B), [y:8,x:16], Partial: ), %input_ids) // Dist(i32[1], (B), [1], ) name='input_ids.reshard15'
  %1   = T.Call(@kernel_embedding_decode_9c2800d81a0929bf, %0, T.ConstAssetRef(Dist(bf16<4>[151936,512], (B,S(C(4)@[0,1])), [151936,4@y@x], ), Recipe: 'constant_recipe_0000', Output: 'token_embedding.packed_weight.reshard16')) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='token_embedding.packed'
  %2   = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %1) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='token_embedding.packed.reshard17'
  %3   = T.Call(@kernel_nn_rotary_embedding_semantic_nn_rotary_embedding_94f1f085ce07d90d, %2, %paged_attention_kv_cache) // (Dist(bf16<2,8>[1,1,8], (B,B,B), [1,1,8], ), Dist(bf16<2,8>[1,1,8], (B,B,B), [1,1,8], )) !read<paged_attention_kv_cache> name='main_rotary_embedding.packed'
  %4   = T.BufferView(AliasKind: 'vector_reinterpret', %2) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_0_decode_layer_call.arg0.boundary_pack'
  %5   = T.GetItem(%3, 0) // Dist(bf16<2,8>[1,1,8], (B,B,B), [1,1,8], ) name='layer_0_decode_layer_call.arg15.boundary_pack'
  %6   = T.GetItem(%3, 1) // Dist(bf16<2,8>[1,1,8], (B,B,B), [1,1,8], ) name='layer_0_decode_layer_call.arg16.boundary_pack'
  %7   = T.Call(@kernel_norm_stats_local_6f038d0656f1ea04, %4) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_0_decode_layer_call.norm_stats'
  %8   = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %4) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_0_decode_layer_call.arg0.distributed_boundary_reshard'
  %9   = T.Call(@decode_layer, %8, %paged_attention_kv_cache, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0005', Output: 'layer_0_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0006', Output: 'layer_0_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0003', Output: 'layer_0_decode_layer_call.arg6.boundary_pack.reshard34'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0004', Output: 'layer_0_decode_layer_call.arg7.boundary_pack.reshard35'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0007', Output: 'layer_0_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0008', Output: 'layer_0_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0009', Output: 'layer_0_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0010', Output: 'layer_0_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 0), T.Const(bool[] : False), %5, %6, %7) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_0_decode_layer_call'
  %10  = T.GetItem(%9, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_0_decoder_output.boundary_packed.distributed_boundary_abi'
  %11  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %10) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_0_decoder_output.boundary_packed'
  %12  = T.GetItem(%9, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_0_updated_state'
  %13  = T.GetItem(%9, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_1_decode_layer_call.norm_stats'
  %14  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %11) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_1_decode_layer_call.arg0.distributed_boundary_reshard'
  %15  = T.Call(@decode_layer, %14, %12, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0013', Output: 'layer_1_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0014', Output: 'layer_1_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0011', Output: 'layer_1_decode_layer_call.arg6.boundary_pack.reshard44'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0012', Output: 'layer_1_decode_layer_call.arg7.boundary_pack.reshard45'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0015', Output: 'layer_1_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0016', Output: 'layer_1_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0017', Output: 'layer_1_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0018', Output: 'layer_1_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 1), T.Const(bool[] : False), %5, %6, %13) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_1_decode_layer_call'
  %16  = T.GetItem(%15, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_1_decoder_output.boundary_packed.distributed_boundary_abi'
  %17  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %16) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_1_decoder_output.boundary_packed'
  %18  = T.GetItem(%15, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_1_updated_state'
  %19  = T.GetItem(%15, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_2_decode_layer_call.norm_stats'
  %20  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %17) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_2_decode_layer_call.arg0.distributed_boundary_reshard'
  %21  = T.Call(@decode_layer, %20, %18, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0021', Output: 'layer_2_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0022', Output: 'layer_2_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0019', Output: 'layer_2_decode_layer_call.arg6.boundary_pack.reshard54'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0020', Output: 'layer_2_decode_layer_call.arg7.boundary_pack.reshard55'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0023', Output: 'layer_2_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0024', Output: 'layer_2_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0025', Output: 'layer_2_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0026', Output: 'layer_2_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 2), T.Const(bool[] : False), %5, %6, %19) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_2_decode_layer_call'
  %22  = T.GetItem(%21, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_2_decoder_output.boundary_packed.distributed_boundary_abi'
  %23  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %22) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_2_decoder_output.boundary_packed'
  %24  = T.GetItem(%21, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_2_updated_state'
  %25  = T.GetItem(%21, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_3_decode_layer_call.norm_stats'
  %26  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %23) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_3_decode_layer_call.arg0.distributed_boundary_reshard'
  %27  = T.Call(@decode_layer, %26, %24, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0029', Output: 'layer_3_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0030', Output: 'layer_3_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0027', Output: 'layer_3_decode_layer_call.arg6.boundary_pack.reshard64'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0028', Output: 'layer_3_decode_layer_call.arg7.boundary_pack.reshard65'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0031', Output: 'layer_3_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0032', Output: 'layer_3_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0033', Output: 'layer_3_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0034', Output: 'layer_3_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 3), T.Const(bool[] : False), %5, %6, %25) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_3_decode_layer_call'
  %28  = T.GetItem(%27, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_3_decoder_output.boundary_packed.distributed_boundary_abi'
  %29  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %28) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_3_decoder_output.boundary_packed'
  %30  = T.GetItem(%27, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_3_updated_state'
  %31  = T.GetItem(%27, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_4_decode_layer_call.norm_stats'
  %32  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %29) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_4_decode_layer_call.arg0.distributed_boundary_reshard'
  %33  = T.Call(@decode_layer, %32, %30, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0037', Output: 'layer_4_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0038', Output: 'layer_4_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0035', Output: 'layer_4_decode_layer_call.arg6.boundary_pack.reshard74'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0036', Output: 'layer_4_decode_layer_call.arg7.boundary_pack.reshard75'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0039', Output: 'layer_4_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0040', Output: 'layer_4_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0041', Output: 'layer_4_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0042', Output: 'layer_4_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 4), T.Const(bool[] : False), %5, %6, %31) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_4_decode_layer_call'
  %34  = T.GetItem(%33, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_4_decoder_output.boundary_packed.distributed_boundary_abi'
  %35  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %34) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_4_decoder_output.boundary_packed'
  %36  = T.GetItem(%33, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_4_updated_state'
  %37  = T.GetItem(%33, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_5_decode_layer_call.norm_stats'
  %38  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %35) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_5_decode_layer_call.arg0.distributed_boundary_reshard'
  %39  = T.Call(@decode_layer, %38, %36, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0045', Output: 'layer_5_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0046', Output: 'layer_5_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0043', Output: 'layer_5_decode_layer_call.arg6.boundary_pack.reshard84'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0044', Output: 'layer_5_decode_layer_call.arg7.boundary_pack.reshard85'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0047', Output: 'layer_5_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0048', Output: 'layer_5_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0049', Output: 'layer_5_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0050', Output: 'layer_5_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 5), T.Const(bool[] : False), %5, %6, %37) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_5_decode_layer_call'
  %40  = T.GetItem(%39, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_5_decoder_output.boundary_packed.distributed_boundary_abi'
  %41  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %40) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_5_decoder_output.boundary_packed'
  %42  = T.GetItem(%39, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_5_updated_state'
  %43  = T.GetItem(%39, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_6_decode_layer_call.norm_stats'
  %44  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %41) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_6_decode_layer_call.arg0.distributed_boundary_reshard'
  %45  = T.Call(@decode_layer, %44, %42, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0053', Output: 'layer_6_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0054', Output: 'layer_6_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0051', Output: 'layer_6_decode_layer_call.arg6.boundary_pack.reshard94'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0052', Output: 'layer_6_decode_layer_call.arg7.boundary_pack.reshard95'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0055', Output: 'layer_6_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0056', Output: 'layer_6_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0057', Output: 'layer_6_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0058', Output: 'layer_6_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 6), T.Const(bool[] : False), %5, %6, %43) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_6_decode_layer_call'
  %46  = T.GetItem(%45, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_6_decoder_output.boundary_packed.distributed_boundary_abi'
  %47  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %46) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_6_decoder_output.boundary_packed'
  %48  = T.GetItem(%45, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_6_updated_state'
  %49  = T.GetItem(%45, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_7_decode_layer_call.norm_stats'
  %50  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %47) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_7_decode_layer_call.arg0.distributed_boundary_reshard'
  %51  = T.Call(@decode_layer, %50, %48, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0061', Output: 'layer_7_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0062', Output: 'layer_7_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0059', Output: 'layer_7_decode_layer_call.arg6.boundary_pack.reshard104'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0060', Output: 'layer_7_decode_layer_call.arg7.boundary_pack.reshard105'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0063', Output: 'layer_7_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0064', Output: 'layer_7_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0065', Output: 'layer_7_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0066', Output: 'layer_7_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 7), T.Const(bool[] : False), %5, %6, %49) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_7_decode_layer_call'
  %52  = T.GetItem(%51, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_7_decoder_output.boundary_packed.distributed_boundary_abi'
  %53  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %52) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_7_decoder_output.boundary_packed'
  %54  = T.GetItem(%51, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_7_updated_state'
  %55  = T.GetItem(%51, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_8_decode_layer_call.norm_stats'
  %56  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %53) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_8_decode_layer_call.arg0.distributed_boundary_reshard'
  %57  = T.Call(@decode_layer, %56, %54, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0069', Output: 'layer_8_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0070', Output: 'layer_8_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0067', Output: 'layer_8_decode_layer_call.arg6.boundary_pack.reshard114'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0068', Output: 'layer_8_decode_layer_call.arg7.boundary_pack.reshard115'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0071', Output: 'layer_8_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0072', Output: 'layer_8_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0073', Output: 'layer_8_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0074', Output: 'layer_8_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 8), T.Const(bool[] : False), %5, %6, %55) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_8_decode_layer_call'
  %58  = T.GetItem(%57, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_8_decoder_output.boundary_packed.distributed_boundary_abi'
  %59  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %58) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_8_decoder_output.boundary_packed'
  %60  = T.GetItem(%57, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_8_updated_state'
  %61  = T.GetItem(%57, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_9_decode_layer_call.norm_stats'
  %62  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %59) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_9_decode_layer_call.arg0.distributed_boundary_reshard'
  %63  = T.Call(@decode_layer, %62, %60, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0077', Output: 'layer_9_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0078', Output: 'layer_9_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0075', Output: 'layer_9_decode_layer_call.arg6.boundary_pack.reshard124'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0076', Output: 'layer_9_decode_layer_call.arg7.boundary_pack.reshard125'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0079', Output: 'layer_9_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0080', Output: 'layer_9_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0081', Output: 'layer_9_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0082', Output: 'layer_9_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 9), T.Const(bool[] : False), %5, %6, %61) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_9_decode_layer_call'
  %64  = T.GetItem(%63, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_9_decoder_output.boundary_packed.distributed_boundary_abi'
  %65  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %64) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_9_decoder_output.boundary_packed'
  %66  = T.GetItem(%63, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_9_updated_state'
  %67  = T.GetItem(%63, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_10_decode_layer_call.norm_stats'
  %68  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %65) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_10_decode_layer_call.arg0.distributed_boundary_reshard'
  %69  = T.Call(@decode_layer, %68, %66, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0085', Output: 'layer_10_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0086', Output: 'layer_10_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0083', Output: 'layer_10_decode_layer_call.arg6.boundary_pack.reshard134'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0084', Output: 'layer_10_decode_layer_call.arg7.boundary_pack.reshard135'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0087', Output: 'layer_10_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0088', Output: 'layer_10_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0089', Output: 'layer_10_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0090', Output: 'layer_10_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 10), T.Const(bool[] : False), %5, %6, %67) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_10_decode_layer_call'
  %70  = T.GetItem(%69, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_10_decoder_output.boundary_packed.distributed_boundary_abi'
  %71  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %70) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_10_decoder_output.boundary_packed'
  %72  = T.GetItem(%69, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_10_updated_state'
  %73  = T.GetItem(%69, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_11_decode_layer_call.norm_stats'
  %74  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %71) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_11_decode_layer_call.arg0.distributed_boundary_reshard'
  %75  = T.Call(@decode_layer, %74, %72, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0093', Output: 'layer_11_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0094', Output: 'layer_11_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0091', Output: 'layer_11_decode_layer_call.arg6.boundary_pack.reshard144'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0092', Output: 'layer_11_decode_layer_call.arg7.boundary_pack.reshard145'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0095', Output: 'layer_11_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0096', Output: 'layer_11_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0097', Output: 'layer_11_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0098', Output: 'layer_11_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 11), T.Const(bool[] : False), %5, %6, %73) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_11_decode_layer_call'
  %76  = T.GetItem(%75, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_11_decoder_output.boundary_packed.distributed_boundary_abi'
  %77  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %76) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_11_decoder_output.boundary_packed'
  %78  = T.GetItem(%75, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_11_updated_state'
  %79  = T.GetItem(%75, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_12_decode_layer_call.norm_stats'
  %80  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %77) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_12_decode_layer_call.arg0.distributed_boundary_reshard'
  %81  = T.Call(@decode_layer, %80, %78, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0101', Output: 'layer_12_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0102', Output: 'layer_12_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0099', Output: 'layer_12_decode_layer_call.arg6.boundary_pack.reshard154'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0100', Output: 'layer_12_decode_layer_call.arg7.boundary_pack.reshard155'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0103', Output: 'layer_12_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0104', Output: 'layer_12_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0105', Output: 'layer_12_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0106', Output: 'layer_12_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 12), T.Const(bool[] : False), %5, %6, %79) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_12_decode_layer_call'
  %82  = T.GetItem(%81, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_12_decoder_output.boundary_packed.distributed_boundary_abi'
  %83  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %82) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_12_decoder_output.boundary_packed'
  %84  = T.GetItem(%81, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_12_updated_state'
  %85  = T.GetItem(%81, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_13_decode_layer_call.norm_stats'
  %86  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %83) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_13_decode_layer_call.arg0.distributed_boundary_reshard'
  %87  = T.Call(@decode_layer, %86, %84, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0109', Output: 'layer_13_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0110', Output: 'layer_13_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0107', Output: 'layer_13_decode_layer_call.arg6.boundary_pack.reshard164'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0108', Output: 'layer_13_decode_layer_call.arg7.boundary_pack.reshard165'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0111', Output: 'layer_13_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0112', Output: 'layer_13_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0113', Output: 'layer_13_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0114', Output: 'layer_13_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 13), T.Const(bool[] : False), %5, %6, %85) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_13_decode_layer_call'
  %88  = T.GetItem(%87, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_13_decoder_output.boundary_packed.distributed_boundary_abi'
  %89  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %88) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_13_decoder_output.boundary_packed'
  %90  = T.GetItem(%87, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_13_updated_state'
  %91  = T.GetItem(%87, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_14_decode_layer_call.norm_stats'
  %92  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %89) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_14_decode_layer_call.arg0.distributed_boundary_reshard'
  %93  = T.Call(@decode_layer, %92, %90, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0117', Output: 'layer_14_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0118', Output: 'layer_14_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0115', Output: 'layer_14_decode_layer_call.arg6.boundary_pack.reshard174'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0116', Output: 'layer_14_decode_layer_call.arg7.boundary_pack.reshard175'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0119', Output: 'layer_14_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0120', Output: 'layer_14_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0121', Output: 'layer_14_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0122', Output: 'layer_14_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 14), T.Const(bool[] : False), %5, %6, %91) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_14_decode_layer_call'
  %94  = T.GetItem(%93, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_14_decoder_output.boundary_packed.distributed_boundary_abi'
  %95  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %94) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_14_decoder_output.boundary_packed'
  %96  = T.GetItem(%93, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_14_updated_state'
  %97  = T.GetItem(%93, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_15_decode_layer_call.norm_stats'
  %98  = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %95) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_15_decode_layer_call.arg0.distributed_boundary_reshard'
  %99  = T.Call(@decode_layer, %98, %96, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0125', Output: 'layer_15_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0126', Output: 'layer_15_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0123', Output: 'layer_15_decode_layer_call.arg6.boundary_pack.reshard184'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0124', Output: 'layer_15_decode_layer_call.arg7.boundary_pack.reshard185'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0127', Output: 'layer_15_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0128', Output: 'layer_15_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0129', Output: 'layer_15_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0130', Output: 'layer_15_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 15), T.Const(bool[] : False), %5, %6, %97) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_15_decode_layer_call'
  %100 = T.GetItem(%99, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_15_decoder_output.boundary_packed.distributed_boundary_abi'
  %101 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %100) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_15_decoder_output.boundary_packed'
  %102 = T.GetItem(%99, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_15_updated_state'
  %103 = T.GetItem(%99, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_16_decode_layer_call.norm_stats'
  %104 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %101) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_16_decode_layer_call.arg0.distributed_boundary_reshard'
  %105 = T.Call(@decode_layer, %104, %102, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0133', Output: 'layer_16_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0134', Output: 'layer_16_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0131', Output: 'layer_16_decode_layer_call.arg6.boundary_pack.reshard194'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0132', Output: 'layer_16_decode_layer_call.arg7.boundary_pack.reshard195'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0135', Output: 'layer_16_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0136', Output: 'layer_16_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0137', Output: 'layer_16_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0138', Output: 'layer_16_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 16), T.Const(bool[] : False), %5, %6, %103) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_16_decode_layer_call'
  %106 = T.GetItem(%105, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_16_decoder_output.boundary_packed.distributed_boundary_abi'
  %107 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %106) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_16_decoder_output.boundary_packed'
  %108 = T.GetItem(%105, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_16_updated_state'
  %109 = T.GetItem(%105, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_17_decode_layer_call.norm_stats'
  %110 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %107) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_17_decode_layer_call.arg0.distributed_boundary_reshard'
  %111 = T.Call(@decode_layer, %110, %108, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0141', Output: 'layer_17_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0142', Output: 'layer_17_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0139', Output: 'layer_17_decode_layer_call.arg6.boundary_pack.reshard204'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0140', Output: 'layer_17_decode_layer_call.arg7.boundary_pack.reshard205'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0143', Output: 'layer_17_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0144', Output: 'layer_17_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0145', Output: 'layer_17_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0146', Output: 'layer_17_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 17), T.Const(bool[] : False), %5, %6, %109) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_17_decode_layer_call'
  %112 = T.GetItem(%111, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_17_decoder_output.boundary_packed.distributed_boundary_abi'
  %113 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %112) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_17_decoder_output.boundary_packed'
  %114 = T.GetItem(%111, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_17_updated_state'
  %115 = T.GetItem(%111, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_18_decode_layer_call.norm_stats'
  %116 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %113) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_18_decode_layer_call.arg0.distributed_boundary_reshard'
  %117 = T.Call(@decode_layer, %116, %114, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0149', Output: 'layer_18_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0150', Output: 'layer_18_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0147', Output: 'layer_18_decode_layer_call.arg6.boundary_pack.reshard214'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0148', Output: 'layer_18_decode_layer_call.arg7.boundary_pack.reshard215'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0151', Output: 'layer_18_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0152', Output: 'layer_18_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0153', Output: 'layer_18_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0154', Output: 'layer_18_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 18), T.Const(bool[] : False), %5, %6, %115) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_18_decode_layer_call'
  %118 = T.GetItem(%117, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_18_decoder_output.boundary_packed.distributed_boundary_abi'
  %119 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %118) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_18_decoder_output.boundary_packed'
  %120 = T.GetItem(%117, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_18_updated_state'
  %121 = T.GetItem(%117, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_19_decode_layer_call.norm_stats'
  %122 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %119) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_19_decode_layer_call.arg0.distributed_boundary_reshard'
  %123 = T.Call(@decode_layer, %122, %120, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0157', Output: 'layer_19_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0158', Output: 'layer_19_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0155', Output: 'layer_19_decode_layer_call.arg6.boundary_pack.reshard224'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0156', Output: 'layer_19_decode_layer_call.arg7.boundary_pack.reshard225'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0159', Output: 'layer_19_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0160', Output: 'layer_19_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0161', Output: 'layer_19_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0162', Output: 'layer_19_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 19), T.Const(bool[] : False), %5, %6, %121) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_19_decode_layer_call'
  %124 = T.GetItem(%123, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_19_decoder_output.boundary_packed.distributed_boundary_abi'
  %125 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %124) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_19_decoder_output.boundary_packed'
  %126 = T.GetItem(%123, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_19_updated_state'
  %127 = T.GetItem(%123, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_20_decode_layer_call.norm_stats'
  %128 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %125) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_20_decode_layer_call.arg0.distributed_boundary_reshard'
  %129 = T.Call(@decode_layer, %128, %126, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0165', Output: 'layer_20_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0166', Output: 'layer_20_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0163', Output: 'layer_20_decode_layer_call.arg6.boundary_pack.reshard234'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0164', Output: 'layer_20_decode_layer_call.arg7.boundary_pack.reshard235'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0167', Output: 'layer_20_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0168', Output: 'layer_20_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0169', Output: 'layer_20_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0170', Output: 'layer_20_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 20), T.Const(bool[] : False), %5, %6, %127) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_20_decode_layer_call'
  %130 = T.GetItem(%129, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_20_decoder_output.boundary_packed.distributed_boundary_abi'
  %131 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %130) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_20_decoder_output.boundary_packed'
  %132 = T.GetItem(%129, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_20_updated_state'
  %133 = T.GetItem(%129, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_21_decode_layer_call.norm_stats'
  %134 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %131) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_21_decode_layer_call.arg0.distributed_boundary_reshard'
  %135 = T.Call(@decode_layer, %134, %132, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0173', Output: 'layer_21_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0174', Output: 'layer_21_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0171', Output: 'layer_21_decode_layer_call.arg6.boundary_pack.reshard244'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0172', Output: 'layer_21_decode_layer_call.arg7.boundary_pack.reshard245'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0175', Output: 'layer_21_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0176', Output: 'layer_21_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0177', Output: 'layer_21_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0178', Output: 'layer_21_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 21), T.Const(bool[] : False), %5, %6, %133) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_21_decode_layer_call'
  %136 = T.GetItem(%135, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_21_decoder_output.boundary_packed.distributed_boundary_abi'
  %137 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %136) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_21_decoder_output.boundary_packed'
  %138 = T.GetItem(%135, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_21_updated_state'
  %139 = T.GetItem(%135, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_22_decode_layer_call.norm_stats'
  %140 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %137) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_22_decode_layer_call.arg0.distributed_boundary_reshard'
  %141 = T.Call(@decode_layer, %140, %138, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0181', Output: 'layer_22_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0182', Output: 'layer_22_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0179', Output: 'layer_22_decode_layer_call.arg6.boundary_pack.reshard254'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0180', Output: 'layer_22_decode_layer_call.arg7.boundary_pack.reshard255'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0183', Output: 'layer_22_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0184', Output: 'layer_22_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0185', Output: 'layer_22_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0186', Output: 'layer_22_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 22), T.Const(bool[] : False), %5, %6, %139) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_22_decode_layer_call'
  %142 = T.GetItem(%141, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_22_decoder_output.boundary_packed.distributed_boundary_abi'
  %143 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %142) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_22_decoder_output.boundary_packed'
  %144 = T.GetItem(%141, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_22_updated_state'
  %145 = T.GetItem(%141, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_23_decode_layer_call.norm_stats'
  %146 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %143) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_23_decode_layer_call.arg0.distributed_boundary_reshard'
  %147 = T.Call(@decode_layer, %146, %144, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0189', Output: 'layer_23_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0190', Output: 'layer_23_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0187', Output: 'layer_23_decode_layer_call.arg6.boundary_pack.reshard264'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0188', Output: 'layer_23_decode_layer_call.arg7.boundary_pack.reshard265'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0191', Output: 'layer_23_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0192', Output: 'layer_23_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0193', Output: 'layer_23_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0194', Output: 'layer_23_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 23), T.Const(bool[] : False), %5, %6, %145) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_23_decode_layer_call'
  %148 = T.GetItem(%147, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_23_decoder_output.boundary_packed.distributed_boundary_abi'
  %149 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %148) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_23_decoder_output.boundary_packed'
  %150 = T.GetItem(%147, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_23_updated_state'
  %151 = T.GetItem(%147, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_24_decode_layer_call.norm_stats'
  %152 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %149) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_24_decode_layer_call.arg0.distributed_boundary_reshard'
  %153 = T.Call(@decode_layer, %152, %150, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0197', Output: 'layer_24_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0198', Output: 'layer_24_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0195', Output: 'layer_24_decode_layer_call.arg6.boundary_pack.reshard274'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0196', Output: 'layer_24_decode_layer_call.arg7.boundary_pack.reshard275'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0199', Output: 'layer_24_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0200', Output: 'layer_24_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0201', Output: 'layer_24_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0202', Output: 'layer_24_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 24), T.Const(bool[] : False), %5, %6, %151) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_24_decode_layer_call'
  %154 = T.GetItem(%153, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_24_decoder_output.boundary_packed.distributed_boundary_abi'
  %155 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %154) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_24_decoder_output.boundary_packed'
  %156 = T.GetItem(%153, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_24_updated_state'
  %157 = T.GetItem(%153, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_25_decode_layer_call.norm_stats'
  %158 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %155) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_25_decode_layer_call.arg0.distributed_boundary_reshard'
  %159 = T.Call(@decode_layer, %158, %156, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0205', Output: 'layer_25_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0206', Output: 'layer_25_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0203', Output: 'layer_25_decode_layer_call.arg6.boundary_pack.reshard284'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0204', Output: 'layer_25_decode_layer_call.arg7.boundary_pack.reshard285'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0207', Output: 'layer_25_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0208', Output: 'layer_25_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0209', Output: 'layer_25_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0210', Output: 'layer_25_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 25), T.Const(bool[] : False), %5, %6, %157) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_25_decode_layer_call'
  %160 = T.GetItem(%159, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_25_decoder_output.boundary_packed.distributed_boundary_abi'
  %161 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %160) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_25_decoder_output.boundary_packed'
  %162 = T.GetItem(%159, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_25_updated_state'
  %163 = T.GetItem(%159, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_26_decode_layer_call.norm_stats'
  %164 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %161) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_26_decode_layer_call.arg0.distributed_boundary_reshard'
  %165 = T.Call(@decode_layer, %164, %162, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0213', Output: 'layer_26_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0214', Output: 'layer_26_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0211', Output: 'layer_26_decode_layer_call.arg6.boundary_pack.reshard294'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0212', Output: 'layer_26_decode_layer_call.arg7.boundary_pack.reshard295'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0215', Output: 'layer_26_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0216', Output: 'layer_26_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0217', Output: 'layer_26_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0218', Output: 'layer_26_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 26), T.Const(bool[] : False), %5, %6, %163) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_26_decode_layer_call'
  %166 = T.GetItem(%165, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_26_decoder_output.boundary_packed.distributed_boundary_abi'
  %167 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,B), [y:8,x:16], Partial: ), %166) // Dist(f32<4>[1,512], (B,B), [1,512], ) name='layer_26_decoder_output.boundary_packed'
  %168 = T.GetItem(%165, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='layer_26_updated_state'
  %169 = T.GetItem(%165, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='layer_27_decode_layer_call.norm_stats'
  %170 = Distributed.ShardedView(NewType: Dist(f32<4>[1,512], (B,S(C(64)@[0])), [y:8,x:16], Partial: ), %167) // Dist(f32<4>[1,512], (B,S(C(64)@[0])), [1,64@y], ) name='layer_27_decode_layer_call.arg0.distributed_boundary_reshard'
  %171 = T.Call(@decode_layer, %170, %168, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(64)@[0])), [64@y], ), Recipe: 'constant_recipe_0221', Output: 'layer_27_decode_layer_call.arg2.distributed_boundary_reshard'), T.ConstAssetRef(bf16<8,2,8>[128,16,32], Recipe: 'constant_recipe_0222', Output: 'layer_27_decode_layer_call.qkv_fused'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0219', Output: 'layer_27_decode_layer_call.arg6.boundary_pack.reshard304'), T.ConstAssetRef(Dist(bf16<8>[16], (B), [16], ), Recipe: 'constant_recipe_0220', Output: 'layer_27_decode_layer_call.arg7.boundary_pack.reshard305'), T.ConstAssetRef(Dist(bf16<8,2,8>[128,256], (B,S(C(2)@[0,1])), [128,2@y@x], ), Recipe: 'constant_recipe_0223', Output: 'layer_27_decode_layer_call.arg8.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0224', Output: 'layer_27_decode_layer_call.arg9.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16[2048,12288], (B,B), [2048,12288], ), Recipe: 'constant_recipe_0225', Output: 'layer_27_decode_layer_call.arg10.distributed_boundary_reshard'), T.ConstAssetRef(Dist(bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], ), Recipe: 'constant_recipe_0226', Output: 'layer_27_decode_layer_call.arg11.distributed_boundary_reshard'), T.Const(i32[] : 27), T.Const(bool[] : True), %5, %6, %169) // (Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ), &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]}, Dist(f32[1,1,1], (B,B,B), [1,1,1], )) !read_write<paged_attention_kv_cache> name='layer_27_decode_layer_call'
  %172 = T.GetItem(%171, 0) // Dist(f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='layer_27_decoder_output.boundary_packed.distributed_boundary_abi'
  %173 = T.GetItem(%171, 2) // Dist(f32[1,1,1], (B,B,B), [1,1,1], ) name='final_norm.stats'
  %174 = T.Call(@kernel_norm_apply_local_f33fb3dd3b151d1a, %172, %173, T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0227', Output: 'final_norm.vectorized.pack2.reshard311'), T.ConstAssetRef(Dist(bf16<4>[512], (S(C(4)@[0,1])), [4@y@x], ), Recipe: 'constant_recipe_0001', Output: 'decode_layer_input_norm.vectorized.pack3.reshard312')) // Dist(bf16<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], ) name='final_norm.vectorized.compute'
  %175 = Distributed.ShardedView(NewType: Dist(bf16<4>[1,512], (B,B), [y:8,x:16], Partial: ), %174) // Dist(bf16<4>[1,512], (B,B), [1,512], ) name='final_norm.vectorized.compute.reshard313'
  %176 = T.BufferView(AliasKind: 'vector_reinterpret', %175) // Dist(bf16[1,2048], (B,B), [1,2048], ) name='final_norm'
  %177 = T.Call(@kernel_dense_matmul_packed_tensor_descriptor_smem_pipeline_gemv_f25069adf84b6d9b, %176, T.ConstAssetRef(Dist(bf16<8,2,8>[128,18992], (B,S(BC(1)@[0,1])), [128,⌈18992/128⌉@y@x], ), Recipe: 'constant_recipe_0000', Output: 'logits.vectorized.compute.rhs_k_major.physical.reshard314'), %w0, %w0) // Dist(f32<2,4>[1,18992], (B,S(BC(1)@[0,1])), [1,⌈18992/128⌉@y@x], ) name='logits.vectorized.compute'
  %178 = T.BufferView(AliasKind: 'vector_reinterpret', %177) // Dist(f32[1,151936], (B,S(BC(8)@[0,1])), [1,1187@y@x], ) name='logits'
  %179 = T.Call(@kernel_greedy_sample_distributed_argmax_65438e715cd66e9c, %178) // Dist(i32[1], (B), [1], ) name='next_token'
  %180 = T.GetItem(%171, 1) // &paged_attention_kv_cache{kv_caches: bf16<8>[16,28,2,256,8,16], query_start_loc: i32[2], seq_lens: i32[1], slot_mapping: i64[1], block_table: i32[1,16]} name='updated_state'
  %181 = Distributed.ShardedView(NewType: Dist(f32[1,151936], (B,B), [y:8,x:16], Partial: ), %178) // Dist(f32[1,151936], (B,B), [1,151936], ) name='logits.reshard328'
  T.Return(%181, %179, %180)
)

// selections
T.Select('distribution.final_norm', 'distribution.final_norm.operation_type_inference_sbp.mesh_yx_8x16_bb.in_d_b_b.out_d_b_b', origin='agent', policy='r016-fullmodel-retained-distribution')
T.Select('distribution.final_norm.vectorized.compute', 'distribution.final_norm.vectorized.compute.norm_apply_compatible_sbp.mesh_yx_8x16_bb.in_d_b_s_c_h0_1_g4__d_b_b_b__d_s_c_h0_1_g4__d_s_c_h0_1_g4.out_d_b_s_c_h0_1_g4', origin='agent', policy='r016-fullmodel-retained-distribution')
T.Select('distribution.layer_0_decode_layer_call.arg0.boundary_pack', 'distribution.layer_0_decode_layer_call.arg0.boundary_pack.operation_type_inference_sbp.mesh_yx_8x16_bb.in_d_b_b.out_d_b_b', origin='agent', policy='r016-fullmodel-retained-distribution')
T.Select('distribution.layer_0_decode_layer_call.norm_stats', 'distribution.layer_0_decode_layer_call.norm_stats.norm_stats_additive_sbp.mesh_yx_8x16_bb.in_d_b_b.out_d_b_b_b', origin='agent', policy='r016-fullmodel-retained-distribution')
T.Select('distribution.logits', 'distribution.logits.operation_type_inference_sbp.mesh_yx_8x16_bb.in_d_b_s_bc_h0_1_b1.out_d_b_s_bc_h0_1_b8', origin='agent', policy='r016-fullmodel-retained-distribution')
T.Select('distribution.logits.vectorized.compute', 'distribution.logits.vectorized.compute.packed_matmul_operand_sbp.mesh_yx_8x16_bb.in_d_b_b__d_b_s_bc_h0_1_b1__nonetype__nonetype.out_d_b_s_bc_h0_1_b1', origin='agent', policy='r016-fullmodel-retained-distribution')
T.Select('distribution.next_token', 'distribution.next_token.local_argmax_and_global_index_merge.mesh_yx_8x16_bb.in_d_b_s_bc_h0_1_b8.out_d_b', origin='agent', policy='r016-fullmodel-retained-distribution')
T.Select('distribution.token_embedding.packed', 'distribution.token_embedding.packed.embedding_exact_output_sbp.mesh_yx_8x16_bb.in_d_b__d_b_s_c_h0_1_g4.out_d_b_s_c_h0_1_g4', origin='agent', policy='r016-fullmodel-retained-distribution')
T.Select('packing.lm_head', 'packing.k_major_n8_k16', origin='default-policy', policy='pyntt-auto-packing/v2(vector_bytes=16,k_pack=2)')
T.Select('tir.final_norm.vectorized.compute', 'tir.norm_apply.local', origin='agent', policy='r016-fullmodel-GU-island-only')
T.Select('tir.layer_0_decode_layer_call.norm_stats', 'tir.norm_stats.local', origin='agent', policy='r016-fullmodel-GU-island-only')
T.Select('tir.logits.vectorized.compute', 'tir.dense_matmul.packed_tensor_descriptor_table_smem_pipeline_gemv_tn64_bk512', origin='agent', policy='r016-fullmodel-GU-island-only')
T.Select('tir.main_rotary_embedding.packed', 'semantic.nn.rotary_embedding', origin='agent', policy='r016-fullmodel-GU-island-only')
T.Select('tir.next_token', 'tir.greedy_sample.distributed_argmax', origin='agent', policy='r016-fullmodel-GU-island-only')
T.Select('tir.token_embedding.packed', 'tir.embedding.decode', origin='agent', policy='r016-fullmodel-GU-island-only')
T.Select('vectorization.final_norm', 'vectorization.norm_apply.reduction_axis', origin='default-policy', policy='ntt-auto-vectorize/v3(lane_bytes=16,max_axes=1)')
T.Select('vectorization.logits', 'vectorization.matmul.n', origin='default-policy', policy='ntt-auto-vectorize/v3(lane_bytes=16,max_axes=1)')
T.Select('microkernel.kernel_nn_rotary_embedding_semantic_nn_rotary_embedding_94f1f085ce07d90d', 'tir.rotary_embedding.decode', origin='default-policy', policy='pyntt-nvidia-sm90/v31')
T.Select('microkernel.kernel_ntt_packed_qkv_parallel_linear_semantic_ntt_packed_qkv_parallel__e4c2ee4f02f7b3c3', 'tir.qkv_parallel_linear.packed_partial_mma_smem_pipeline', origin='default-policy', policy='pyntt-nvidia-sm90/v31')
T.Select('microkernel.kernel_ntt_gather_reduce_qkv_rope_with_cache_semantic_ntt_gather_reduce_4de2b227526601fc', 'tir.gather_reduce_qkv_rope_with_cache.decode', origin='default-policy', policy='pyntt-nvidia-sm90/v31')
T.Select('microkernel.kernel_ntt_paged_attention_partial_semantic_ntt_paged_attention_partial_499e606b2b12cd57', 'tir.paged_attention_partial.mma_tma_smem_pipeline', origin='default-policy', policy='pyntt-nvidia-sm90/v31')
T.Select('microkernel.kernel_ntt_paged_attention_combine_semantic_ntt_paged_attention_combine_9503a711940aa3e2', 'tir.paged_attention_combine.decode', origin='default-policy', policy='pyntt-nvidia-sm90/v31')
