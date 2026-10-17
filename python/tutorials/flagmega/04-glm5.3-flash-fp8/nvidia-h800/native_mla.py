"""Workload-local, explicit BF16 latent-cache MLA construction."""
from dataclasses import replace

from triton.flagmega import ir as fm
from triton.flagmega.ir.ops.nn._paged_attention_state import paged_attention_state_config_from_type
from triton.flagmega.passes.functions.remove_unused import remove_unused_functions
from triton.flagmega.rules.neutral._utility import make_node


def apply_native_mla(module):
    """Keep cache publication separate from Q absorption/attention/V expansion."""
    prefix = 'decode_mla_attention_sparse'
    projection = module.node_map[prefix+'_self_attn_kv_b_proj_weight_apply']
    attention = module.node_map[prefix+'_dsa_attention']
    assert projection.op in {'math.matmul', 'math.block_scaled_matmul'}
    assert attention.op == 'nn.dsa_paged_attention'
    assert attention.attrs.get('probability_rounding', 'normalized') == 'normalized'
    update = module.node_map[attention.inputs[2]]
    assert update.op == 'nn.update_paged_attention_kv_cache_pair'
    assert tuple(update.attrs['layout']) == ('seq', 'head', 'dim')
    old_config = paged_attention_state_config_from_type(update.type)
    query_type = module.node_map[attention.inputs[0]].type
    heads = query_type.shape[1].fixed_value
    qk_dim = query_type.shape[2].fixed_value*query_type.dtype.lane_count
    compressed_type = module.node_map[projection.inputs[0]].type
    assert compressed_type.dtype == fm.DType.BFLOAT16 and compressed_type.rank == 2
    latent = compressed_type.shape[1].fixed_value
    weight_type = module.node_map[projection.inputs[1]].type
    value_dim = weight_type.shape[0].fixed_value//heads-qk_dim
    assert old_config.num_kv_heads == heads and old_config.head_dim == qk_dim == value_dim
    new_config = replace(old_config, num_kv_heads=1, head_dim=latent)
    old_ref, new_ref = old_config.ref_type, new_config.ref_type

    def retype(kind):
        if kind == old_ref:
            return new_ref
        if isinstance(kind, fm.TupleType):
            return replace(kind, fields=tuple(retype(field) for field in kind.fields))
        return kind

    nodes, mapped = [], {}
    occupied = set(module.node_map)

    def emit(op, name, inputs, attrs, metadata=None, *, replace_existing=False):
        if not replace_existing:
            assert name not in occupied, name
            occupied.add(name)
        node = make_node(op, name, inputs, attrs, metadata or {})
        nodes.append(node)
        return node

    for source in module.nodes:
        inputs = tuple(mapped[key] for key in source.inputs)
        if source.id == update.id:
            compressed = mapped[projection.inputs[0]]
            pair = emit('tensors.concat', source.id+'.latent_pair', (compressed, compressed), {'axis': 1})
            pair = emit('tensors.reshape', source.id+'.latent_heads', (pair,), {'shape': (1, 1, 2*latent)})
            packed = emit('tensors.pack', source.id+'.latent_packed', (pair,),
                {'lanes': (new_config.lanes,), 'axes': (2,)})
            result = emit(source.op, source.id, (packed, *inputs[1:]), source.attrs,
                source.metadata, replace_existing=True)
        elif source.id == attention.id:
            weight = mapped[projection.inputs[1]]
            scale = (mapped[projection.inputs[2]] if projection.op == 'math.block_scaled_matmul' else
                emit('builtin.none', source.id+'.scale_none', (), {}))
            attrs = dict(qk_head_dim=qk_dim, value_head_dim=value_dim, latent_dim=latent,
                weight_block_n=projection.attrs.get('weight_block_n', 128),
                weight_block_k=projection.attrs.get('weight_block_k', 128),
                output_lanes=tuple(query_type.dtype.lanes))
            projected = emit('nn.mla_projection', source.id+'.absorbed_query', (inputs[0], weight, scale),
                {**attrs, 'direction': 'absorb_query'})
            attended = emit(source.op, source.id+'.latent', (projected, *inputs[1:]),
                {**source.attrs, 'hidden_size': heads*latent, 'probability_rounding': 'online_bf16',
                 'index_traversal': 'compact'})
            result = emit('nn.mla_projection', source.id, (attended, weight, scale),
                {**attrs, 'direction': 'expand_value'}, source.metadata, replace_existing=True)
            assert result.type == source.type
        elif source.op in {'builtin.call', 'control.for'} or not source.inputs:
            result = replace(source, inputs=tuple(value.id for value in inputs), type=retype(source.type))
            nodes.append(result)
        else:
            result = emit(source.op, source.id, inputs, source.attrs, source.metadata, replace_existing=True)
        mapped[source.id] = result
    result = replace(module, nodes=tuple(nodes), metadata={**module.metadata,
        'numerical_contract': str(module.metadata['numerical_contract'])+'/latent-mla-bf16-online64',
        'native_mla_contract': {'query_absorption': 'bf16', 'value_expansion': 'bf16',
            'probability_rounding': 'online_bf16', 'token_tile': 64, 'index_traversal': 'compact',
            'cache_num_kv_heads': 1, 'cache_head_dim': latent,
            'state_publication': 'separate_paired_update', 'official_prefix_accepted': False}})
    return fm.verify_module(remove_unused_functions(result))
