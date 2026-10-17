"""Explicit native arithmetic profile for complete GLM TP4 reproduction.

The default nncase contract is unchanged. The reference is pinned vLLM 0.31.0;
full independent greedy-prefix acceptance remains required.
"""
from dataclasses import replace

from triton.flagmega.ir import DType, TupleType, tensor_type, verify_module
from triton.flagmega.ir.ops.nn.sparse_experts import SparseExperts
from triton.flagmega.ir.ops.nn.sparse_experts_gate_up import SparseExpertsGateUp
from triton.flagmega.passes.functions import function_nodes
from triton.flagmega.rules.neutral._utility import make_node

PROFILE = 'glm5-vllm-0.31-tp4'


def _keep_raw_kda_decay(module):
    """Move exp(A_log) from the constant island into its owning KDA op."""
    raw = {}
    for node in module.nodes:
        if node.op != 'math.exp':
            continue
        value = module.node_map[node.inputs[0]]
        origin = value
        while origin.op in {'tensors.cast', 'tensors.reshape'}:
            origin = module.node_map[origin.inputs[0]]
        if origin.op != 'builtin.weight' or not str(origin.attrs.get('key', '')).endswith('.self_attn.A_log'):
            continue
        assert value.type == node.type and value.type.dtype == DType.FLOAT32, node.id
        raw[node.id] = value.id
    if not raw:
        raise ValueError('The native KDA profile requires explicit checkpoint A_log maps.')
    nodes = []
    consumers = 0
    for node in module.nodes:
        if node.id in raw:
            continue
        node = replace(node, inputs=tuple(raw.get(value, value) for value in node.inputs))
        if node.op in {'nn.kda_gated_recurrent_core', 'nn.kda_gated_step'}:
            node = replace(node, attrs={**node.attrs, 'decay_input': 'log_rate'})
            consumers += 1
        nodes.append(node)
    if not consumers:
        raise ValueError('Raw A_log maps have no gated KDA consumer.')
    functions = tuple(replace(function, outputs=tuple(raw.get(value, value) for value in function.outputs))
                      for function in module.functions)
    return verify_module(replace(module, nodes=tuple(nodes), functions=functions,
        metadata={**module.metadata, 'native_kda_gate': {
            'decay_input': 'log_rate', 'runtime_exp': True, 'source_maps': len(raw)}}))


def _apply_projection_profile(module):
    assert module.stage == 'imported'
    assert module.metadata.get('numerical_contract', 'nncase') == 'nncase'
    decoders = {f.name for f in module.functions if f.name.startswith('decode_')}
    hidden_parameters = {module.function_map[name].parameters[0] for name in decoders}
    nodes, mapped = [], {}

    def emit(op, name, inputs=(), attrs=None, metadata=None):
        result = make_node(op, name, inputs, attrs or {}, metadata or {})
        nodes.append(result)
        return result

    def cast(value, dtype, name):
        return value if value.type.dtype == DType(dtype) else emit('tensors.cast', name, (value,), {'dtype': dtype})

    def reshape(value, shape, name):
        return emit('tensors.reshape', name, (value,), {'shape': tuple(shape)})

    def slice_(value, axis, begin, end, name):
        return emit('tensors.slice', name, (value,), {'starts': (begin,), 'ends': (end,), 'axes': (axis,)})

    def permute(value, axes, name):
        return emit('tensors.permute', name, (value,), {'axes': axes})

    def sum_partitions(value, name):
        wide = cast(value, 'float32', name + '.wide')
        total = emit('math.reduce_sum', name + '.sum', (wide,), {'axes': (0,), 'keep_dims': False})
        return cast(total, 'bfloat16', name)

    def row_projection(op, operands, attrs, name):
        """Each TP rank stores its local BF16 dot before the FP32 collective."""
        value, weight, *scales = operands
        k = value.type.shape[-1].fixed_value
        assert k % 4 == 0
        span = k // 4
        pieces = []
        for p in range(4):
            base = name + f'.partition{p}'
            local_q = slice_(value, 1, p * span, (p + 1) * span, base + '.input')
            local_w = slice_(weight, 1, p * span, (p + 1) * span, base + '.weight')
            inputs = (local_q, local_w)
            if scales:
                bk = attrs['weight_block_k']
                assert span % bk == 0
                local_s = slice_(scales[0], 1, p * span // bk, (p + 1) * span // bk, base + '.scale')
                inputs += (local_s,)
            local = emit(op, base + '.dot', inputs, attrs)
            pieces.append(reshape(local, (1, *[d.fixed_value for d in local.type.shape]), base + '.axis'))
        stacked = emit('tensors.concat', name + '.partitions', pieces, {'axis': 0})
        return sum_partitions(stacked, name)

    def routed(source, inputs):
        values = {p.name: p.read(inputs) for p in SparseExperts.input_parameters}
        dispatched = emit('nn.sparse_experts_dispatch', source.id + '.dispatch',
                          (values['q'], values['router_expert_ids']))
        values['dispatched'] = dispatched
        gate = emit('nn.sparse_experts_gate_up', source.id + '.gate_up',
                    tuple(values[p.name] for p in SparseExpertsGateUp.input_parameters),
                    {**{k: source.attrs[k] for k in ('gate_clamp_max', 'up_clamp', 'weight_block_n', 'weight_block_k')},
                     'round_projections': True, 'round_activation': True,
                     'activation_quantization': 'dynamic_fp8_ue8m0'})
        t, r, width = [d.fixed_value for d in gate.type.shape]
        e, h, k = [d.fixed_value for d in values['down_weight'].type.shape]
        assert width == k and k % 4 == 0
        act = reshape(gate, (t, r, 4, k // 4), source.id + '.activation_partitions')
        w = reshape(values['down_weight'], (e, h, 4, k // 4), source.id + '.weight_partition_axis')
        w = permute(w, (0, 2, 1, 3), source.id + '.weight_partitions')
        ng, kg = [d.fixed_value for d in values['down_proj_scale'].type.shape[-2:]]
        assert kg % 4 == 0
        s = reshape(values['down_proj_scale'], (e, ng, 4, kg // 4), source.id + '.scale_partition_axis')
        s = permute(s, (0, 2, 1, 3), source.id + '.scale_partitions')
        down = emit('nn.partitioned_sparse_experts_down', source.id + '.down',
                    (act, values['router_expert_ids'], values['down_input_scale'], w, s),
                    {'weight_block_n': source.attrs['weight_block_n'],
                     'weight_block_k': source.attrs['weight_block_k'],
                     'activation_quantization': 'dynamic_fp8_ue8m0', 'round_projection': False})
        down = reshape(down, (4 * t, r, h), source.id + '.down_flat')
        coeff = emit('tensors.broadcast_to', source.id + '.coeff_partitions',
                     (values['router_expert_weights'],), {'shape': (4, t, r)})
        coeff = reshape(coeff, (4 * t, r), source.id + '.coeff_flat')
        local = emit('nn.sparse_experts_combine', source.id + '.combine', (down, coeff),
                     {'output_dtype': 'bfloat16', 'round_weighted_output': True})
        return reshape(local, (4, t, h), source.id)

    def shared(source, inputs):
        v = {p.name: p.read(inputs) for p in SparseExperts.input_parameters}
        weights = {}
        for stage in ('gate', 'up', 'down'):
            for field in ('weight', 'proj_scale'):
                value = v[stage + '_' + field]
                weights[stage + '_' + field] = reshape(value, tuple(d.fixed_value for d in value.type.shape[1:]),
                                                        source.id + '.' + stage + '_' + field)
        attrs = {'weight_block_n': source.attrs['weight_block_n'], 'weight_block_k': source.attrs['weight_block_k'],
                 'activation_quantization': 'dynamic_fp8_sm90'}
        gate = emit('nn.matmul_glu', source.id + '.gate_up',
                    (v['q'], weights['gate_weight'], weights['up_weight'],
                     weights['gate_proj_scale'], weights['up_proj_scale']),
                    {**attrs, 'activation': 'silu', 'gate_clamp_max': source.attrs['gate_clamp_max'],
                     'up_clamp': source.attrs['up_clamp']})
        # Keep the shared local BF16 output beside routed output until their
        # local BF16 add; the cross-rank reduction follows that add.
        t, k = [d.fixed_value for d in gate.type.shape]
        h = weights['down_weight'].type.shape[0].fixed_value
        span = k // 4
        assert k % 4 == 0 and span % attrs['weight_block_k'] == 0
        parts = []
        for p in range(4):
            prefix = source.id + f'.partition{p}'
            q = slice_(gate, 1, p * span, (p + 1) * span, prefix + '.input')
            w = slice_(weights['down_weight'], 1, p * span, (p + 1) * span, prefix + '.weight')
            bk = attrs['weight_block_k']
            scale = slice_(weights['down_proj_scale'], 1, p * span // bk, (p + 1) * span // bk, prefix + '.scale')
            local = emit('math.block_scaled_matmul', prefix + '.dot', (q, w, scale), attrs)
            parts.append(reshape(local, (1, t, h), prefix + '.axis'))
        return emit('tensors.concat', source.id, parts, {'axis': 0})

    # Only the FP8 linear backend requantizes these dense/shared MLP weights.
    # MLA loads BF16 dequantized weights; routed expert tensors keep their
    # separate MoE checkpoint/kernel contract.
    weight_nodes = {n.attrs.get('key'): n for n in module.nodes if n.op == 'builtin.weight'}
    requant_pairs, reverse_pairs = {}, {}
    for key, weight in weight_nodes.items():
        if not isinstance(key, str) or weight.type.dtype != DType.FLOAT8_E4M3FN:
            continue
        if '.mlp.' not in key or '.mlp.experts.' in key:
            continue
        scale = weight_nodes.get(key + '_scale_inv')
        if scale is not None:
            requant_pairs[weight.id] = scale
            reverse_pairs[scale.id] = weight

    for source in module.nodes:
        if source.id in mapped:
            continue
        inputs = tuple(mapped[name] for name in source.inputs)
        if source.id in requant_pairs or source.id in reverse_pairs:
            weight = source if source.id in requant_pairs else reverse_pairs[source.id]
            scale = requant_pairs[weight.id]
            raw_w, raw_s = replace(weight, id=weight.id + '.checkpoint'), replace(scale, id=scale.id + '.checkpoint')
            nodes.extend((raw_w, raw_s))
            pair = emit('nn.requantize_block_scaled_fp8', weight.id + '.requantize', (raw_w, raw_s),
                        {'block_n': 128, 'block_k': 128, 'scale_division': 'reciprocal_multiply'})
            for i, n in enumerate((weight, scale)):
                mapped[n.id] = emit('builtin.get_item', n.id, (pair,), {'index': i})
            continue
        if source.id in hidden_parameters:
            result = replace(source, type=tensor_type('float32', source.type.shape))
            nodes.append(result)
        elif source.op == 'builtin.call' and source.attrs.get('callee') in decoders:
            hidden = cast(inputs[0], 'float32', source.id + '.hidden_wide')
            result = replace(source, inputs=(hidden.id, *(v.id for v in inputs[1:])),
                             type=TupleType((hidden.type, *source.type.fields[1:])))
            nodes.append(result)
        elif source.op == 'nn.mhc_pre':
            result = emit(source.op, source.id, inputs,
                          {**source.attrs, 'collapsed_dtype': 'float32', 'collapse_input_dtype': 'bfloat16',
                           'collapse_reduction': 'sequential_fma'})
        elif source.op == 'nn.mhc_post':
            residual = cast(inputs[1], 'bfloat16', source.id + '.residual_bf16')
            result = emit(source.op, source.id, (inputs[0], residual, *inputs[2:]),
                          {**source.attrs, 'output_dtype': 'float32', 'accumulation_mode': 'value_fma_first'})
        elif source.op == 'nn.staged_routing':
            result = emit(source.op, source.id, inputs,
                          {**source.attrs, 'arithmetic': 'tanh_scaled_reciprocal'}, source.metadata)
        elif source.op == 'nn.norm_apply' and source.id in {
                name + suffix for name in decoders for suffix in ('_input_norm', '_post_norm')}:
            numerator = cast(inputs[0], 'bfloat16', source.id + '.numerator_bf16')
            result = emit(source.op, source.id, (numerator, *inputs[1:]), source.attrs)
        elif source.op == 'nn.matmul_glu':
            result = emit(source.op, source.id, inputs,
                          {**source.attrs, 'activation_quantization': 'dynamic_fp8_sm90'})
        elif source.op == 'math.block_scaled_matmul':
            mla = 'self_attn' in source.id
            attrs = {**source.attrs, **({'activation_quantization': 'none', 'weight_dequant_dtype': 'bfloat16'}
                                       if mla else {'activation_quantization': 'dynamic_fp8_sm90'})}
            if 'o_proj' in source.id or 'dense_mlp_down' in source.id:
                result = row_projection(source.op, inputs, attrs, source.id)
            else:
                result = emit(source.op, source.id, inputs, attrs)
        elif source.op == 'math.matmul' and 'self_attn_o_proj_weight_apply' in source.id:
            result = row_projection(source.op, inputs, source.attrs, source.id)
        elif source.op == 'nn.sparse_experts':
            result = shared(source, inputs) if 'shared_experts' in source.id else routed(source, inputs)
        elif source.op == 'math.add' and source.id.endswith('_moe_output'):
            local = emit(source.op, source.id + '.local', inputs, source.attrs)
            result = sum_partitions(local, source.id)
        elif source.id == 'collapse_head_streams':
            last = cast(inputs[0], 'bfloat16', source.id + '.last_streams_bf16')
            result = emit(source.op, source.id, (last,), source.attrs)
        elif source.id == 'logits' and source.op == 'math.matmul':
            bf16 = emit(source.op, source.id + '.bf16', inputs, {**source.attrs, 'output_data_type': 'bfloat16'})
            result = cast(bf16, 'float32', source.id)
        elif not source.inputs:
            result = source
            nodes.append(result)
        else:
            result = emit(source.op, source.id, inputs, source.attrs, source.metadata)
        mapped[source.id] = result
    result = replace(module, nodes=tuple(nodes), metadata={**module.metadata,
                     'numerical_contract': PROFILE, 'numerical_profile_reference': {
                         'vllm': '0.31.0', 'tensor_parallel_size': 4, 'enforce_eager': True,
                         'acceptance': 'independent greedy token prefix; not tensor equality',
                         'status': 'arithmetic experiment, full-model acceptance pending'}})
    return verify_module(result)


def _specialize_initial_mhc(module):
    """Express the initial and repeated pre boundaries outside the shared decoder."""
    first = module.node_map['layer_0_call']
    source = module.function_map[first.attrs['callee']]
    projection = module.node_map[source.name + '_attn_hyper_connection']
    assert projection.op == 'nn.mhc_pre'
    region = tuple(function_nodes(module, source))
    fields = {int(node.attrs['index']): node for node in region
              if node.op == 'builtin.get_item' and node.inputs == (projection.id,)}
    assert set(fields) == {0, 1, 2}
    assert {node.id for node in module.nodes if projection.id in node.inputs} == {node.id for node in fields.values()}
    callers = {node.id: node for node in module.nodes
               if node.op == 'builtin.call' and node.attrs['callee'] == source.name}
    assert first.id in callers
    nodes = []
    field_ids = {node.id for node in fields.values()}
    for node in module.nodes:
        if node.id == projection.id:
            continue
        if node.id in field_ids:
            node = replace(node, op='builtin.var', inputs=(), attrs={'name': node.id},
                           metadata={**node.metadata, 'function_parameter': source.name})
        if node.id in callers:
            bindings = dict(zip(source.parameters, node.inputs, strict=True))
            operands = tuple(module.node_map[bindings[value]] for value in projection.inputs)
            attrs = dict(projection.attrs)
            if node.id == first.id:
                attrs.update(projection_reduction='contiguous_tf32', projection_splits=64)
            pre = make_node('nn.mhc_pre', node.id + '.attn_pre', operands, attrs, projection.metadata)
            outputs = tuple(make_node('builtin.get_item', pre.id + f'.{i}', (pre,), {'index': i}, {}) for i in range(3))
            nodes.extend((pre, *outputs))
            node = replace(node, inputs=(*node.inputs, *(value.id for value in outputs)))
        nodes.append(node)
    function = replace(source, parameters=(*source.parameters, *(fields[i].id for i in range(3))))
    return verify_module(replace(module, nodes=tuple(nodes),
        functions=tuple(function if value.name == source.name else value for value in module.functions),
        metadata={**module.metadata, 'initial_hyperconnection_projection': 'contiguous_tf32/split64',
                  'dense_attention_pre_boundary': 'caller'}))


def apply_native_profile(module):
    """Construct the explicit single-token prefill/decode arithmetic experiment.

    This opt-in profile keeps the nncase importer and strict reference unchanged.
    Full official greedy-prefix acceptance is still required after compilation.
    """
    if module.stage != 'imported' or module.metadata.get('numerical_contract', 'nncase') != 'nncase':
        raise ValueError('The native arithmetic profile requires a fresh nncase import.')
    if module.metadata.get('num_hidden_layers') != 45 or module.metadata.get('scope') != 'text-decoder':
        raise ValueError('This profile requires the complete 45-layer GLM-5.3 text decoder.')
    source = _keep_raw_kda_decay(_apply_projection_profile(module))
    assert source.metadata['native_kda_gate']['source_maps'] == 34
    regions = {function.name: tuple(function_nodes(source, function)) for function in source.functions}
    linear = {name for name, nodes in regions.items()
              if any(node.op == 'nn.kda_gated_recurrent_core' for node in nodes)}
    if len(linear) != 2:
        raise ValueError(f'Expected the dense and sparse KDA functions, got {sorted(linear)}.')
    phases = {name: f'{name}_request_prefill' for name in (*sorted(linear), source.entry)}
    owner = {node.id: name for name, nodes in regions.items() for node in nodes}
    prototype = source.node_map[source.function_map[source.entry].parameters[0]]
    phase_nodes = {name: replace(prototype, id=identity, type=tensor_type('bool', ()),
                                 attrs={'name': 'request_prefill'}, metadata={'function_parameter': name})
                   for name, identity in phases.items()}
    nodes = list(phase_nodes.values())
    for node in source.nodes:
        if node.op == 'nn.kda_gated_recurrent_core':
            node = make_node('nn.kda_gated_step', node.id,
                             tuple(source.node_map[value] for value in node.inputs) + (phase_nodes[owner[node.id]],),
                             {key: value for key, value in node.attrs.items() if key != 'arithmetic'}, node.metadata)
        elif node.op == 'nn.mhc_pre':
            node = make_node(node.op, node.id, tuple(source.node_map[value] for value in node.inputs),
                             {**node.attrs, 'projection_reduction': 'stream_fma_128', 'projection_splits': 8}, node.metadata)
        elif node.op == 'builtin.call' and node.attrs.get('callee') in linear:
            node = replace(node, inputs=(*node.inputs, phases[owner[node.id]]))
        nodes.append(node)
    functions = tuple(replace(function, parameters=(*function.parameters, phases[function.name]))
                      if function.name in phases else function for function in source.functions)
    from native_mla import apply_native_mla
    return apply_native_mla(_specialize_initial_mhc(verify_module(replace(source, nodes=tuple(nodes), functions=functions,
        metadata={**source.metadata, 'floating_point_contract': 'strict', 'execution_phase': 'request',
                  'numerical_contract': PROFILE + '/mhc-stream-fma-128-s8/request-phase',
                  'composite_abi': {**source.metadata['composite_abi'], 'prefill_input': phases[source.entry]},
                  'numerical_profile_reference': {**source.metadata['numerical_profile_reference'],
                      'status': 'experimental; full official prefix acceptance pending'}}))))
