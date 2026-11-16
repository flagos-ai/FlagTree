// FlagMega high_level stage=selected_tir_variants semantic_hash=7edde262d2940a2199210c9d6247a6427027a23e3612990ebe229d603ec40c96
%decode_layer.after_gu = fn(%decode_layer.after_gu.arg0: {bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, %decode_layer.after_gu.arg1: {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, %decode_layer.after_gu.arg2: {bf16[1,6144], (B,B), [1,6144], }): // ({bf16<8,2,8>[384,256], (B,S(C(2)@[0,1])), [384,2@y@x], }, {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], }, {bf16[1,6144], (B,B), [1,6144], }) -> ({f32<4>[1,512], (B,B), [1,512], }, {f32[1,1,1], (B,B,B), [1,1,1], })
{
  weights {
    // display grouping only; not an execution schedule
    %w0 = None(): // none name='decode_layer_qkv_none'
  }
  // compute
  %0 = NTT.PackedMatMul(FusedReduce: False, OutputDataType: 'bfloat16', RhsLayout: 'k_major', %decode_layer.after_gu.arg2, %decode_layer.after_gu.arg0, %w0, %w0): // {bf16<8>[1,256], (B,S(C(2)@[0,1])), [1,2@y@x], } name='decode_layer_mlp_down.vectorized.compute'
  %1 = Tensors.Bitcast(Dtype: 'bfloat16', %0): // {bf16[1,2048], (B,S(C(16)@[0,1])), [1,16@y@x], } name='decode_layer_output.projection_wide.cast_input'
  %2 = Math.VectorizedBinary(BinaryOp: 'add', PreOps: {rhs: fusion(value: bf16[1,2048]) => Tensors.Cast(value, Dtype: 'float32', RoundingContract: 'preserve')}, RhsPack: {axes: [1], lanes: [4]}, %decode_layer.after_gu.arg1, %1): // {f32<4>[1,512], (B,S(C(4)@[0,1])), [1,4@y@x], } name='decode_layer_output.vectorized.compute'
  %3 = Distributed.ShardedView(NewType: f32<4>[1,512], (B,B), [y:8,x:16], Partial: , %2): // {f32<4>[1,512], (B,B), [1,512], } name='decode_layer_output.vectorized.compute.reshard327'
  %4 = NN.NormStats(Axis: 1, UseMean: False, %3): // {f32[1,1,1], (B,B,B), [1,1,1], } name='decode_layer.output_norm_stats'
  return (%3, %4)
}

// selections
select packing.decode_layer_mlp_down = packing.k_major_n8_k16 // kind=packing, origin=default-policy
select distribution.decode_layer_mlp_down.vectorized.compute = distribution.decode_layer_mlp_down.vectorized.compute.packed_matmul_operand_sbp.mesh_yx_8x16_bb.in_d_b_b__d_b_s_c_h0_1_g2__nonetype__nonetype.out_d_b_s_c_h0_1_g2 // kind=distribution, origin=agent
select distribution.decode_layer_output.projection_wide.cast_input = distribution.decode_layer_output.projection_wide.cast_input.operation_type_inference_sbp.mesh_yx_8x16_bb.in_d_b_s_c_h0_1_g2.out_d_b_s_c_h0_1_g16 // kind=distribution, origin=agent
select distribution.decode_layer_output.vectorized.compute = distribution.decode_layer_output.vectorized.compute.operation_type_inference_sbp.mesh_yx_8x16_bb.in_d_b_s_c_h0_1_g4__d_b_s_c_h0_1_g16.out_d_b_s_c_h0_1_g4 // kind=distribution, origin=agent
select distribution.decode_layer.output_norm_stats = distribution.decode_layer.output_norm_stats.norm_stats_additive_sbp.mesh_yx_8x16_bb.in_d_b_b.out_d_b_b_b // kind=distribution, origin=agent
select tir.decode_layer_mlp_down.vectorized.compute = tir.dense_matmul.packed_tensor_descriptor_smem_pipeline_gemv // kind=tir, origin=default-policy
select tir.decode_layer_output.vectorized.compute = tir.elementwise.add // kind=tir, origin=default-policy
select tir.decode_layer.output_norm_stats = tir.norm_stats.local // kind=tir, origin=default-policy
