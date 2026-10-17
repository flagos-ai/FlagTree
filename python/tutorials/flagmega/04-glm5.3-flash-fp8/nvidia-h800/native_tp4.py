"""Workload-local complete-K and FP8 MoE choices for the explicit native profile."""
import hashlib
import json
from dataclasses import replace
from math import prod
from pathlib import Path

from tp4_distribution import (
    OUTPUT_SPLITS,
    Tp4DistributionTarget,
    _moe_contracts,
    select_moe_layout,
    select_output_split,
)
from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import AutoDistributedPass, solve_search_graph
from triton.flagmega.passes.auto_distributed.materializer import distribution_selection_state
from triton.flagmega.passes.auto_distributed.proposal_analysis import remember_proposal


class NativeTp4DistributionTarget(Tp4DistributionTarget):
    def propose_distribution(self, module):
        trial = self.evidence_dir
        if trial is None:
            raise ValueError('Native TP4 selection requires an evidence directory.')
        graph = AutoDistributedPass._build_graph(module, self)
        projection_ops={'nn.embedding','ntt.packed_matmul','nn.matmul_glu','nn.packed_matmul_glu',
                        'math.packed_block_scaled_matmul','math.block_scaled_matmul'}
        fixed = {name:select_output_split(graph.bucket_map[name], axes).id
                 for name,axes in OUTPUT_SPLITS.items() if name in graph.bucket_map
                 and module.node_map[name].op in projection_ops}
        # Four explicit local projections are a diagnostic IR decomposition;
        # output sharding preserves each partition's complete reduction.
        for name,bucket in graph.bucket_map.items():
            if '.partition' in name and '.dot' in name and module.node_map[name].op in projection_ops:
                fixed[name] = select_output_split(bucket, (0,1,2)).id
            if name.endswith('shared_experts.gate_up') and module.node_map[name].op in (
                    'nn.matmul_glu','nn.packed_matmul_glu'):
                fixed[name] = select_output_split(bucket, (0,1,2)).id
        # Preserve the complete dot on one owner; output sharding keeps the
        # selected BF16 Tensor Core accumulation intact.
        axes_order=((0,1,2),(1,2),(0,2),(0,1),(2,),(1,),(0,))
        for name,bucket in graph.bucket_map.items():
            node=module.node_map[name]
            if node.op!='ntt.packed_matmul':continue
            lhs=fm.logical_type(module.node_map[node.inputs[0]].type)
            if lhs.dtype is not fm.DType.BFLOAT16:continue
            groups=fm.logical_type(node.type).shape[-1].fixed_value
            matches=[]
            for axes in axes_order:
                if prod(graph.placement.hierarchy[a] for a in axes)>groups:continue
                try:candidate=select_output_split(bucket,axes)
                except ValueError:continue
                matches.append(candidate)
                break
            if not matches:
                def broadcast(t):
                    return (not isinstance(t,fm.DistributedType) or
                        t.partial is None and t.exclusive is None and all(p==fm.SBP.broadcast() for p in t.axis_policies))
                matches=[c for c in bucket.candidates if broadcast(c.return_type) and all(broadcast(t) for t in c.input_types)]
            assert len(matches)==1,(name,groups,[c.id for c in matches])
            fixed[name]=matches[0].id
        b = fm.SBP.broadcast()
        partition = fm.SBP.split_contiguous((0,), 1)
        route = fm.SBP.split_contiguous((1,), 1)
        hidden = fm.SBP.split_contiguous((2,), 256)
        contracts = _moe_contracts()
        contracts['routed_experts.down'] = (
            (((b,route,partition,b), None), ((b,route),None), ((b,b),None),
             ((b,partition,hidden,b),None), ((b,b,b,b),None)),
            ((partition,b,route,hidden),None))
        for function in ('decode_linear_attention_sparse','decode_mla_attention_sparse'):
            for suffix,contract in contracts.items():
                if not suffix.startswith('routed_'):
                    continue
                name = function+'_'+suffix
                bucket = graph.bucket_map[name]
                try:
                    fixed[name] = select_moe_layout(bucket,contract).id
                except ValueError:
                    (trial/'failed-bucket.txt').write_text('\n'.join(
                        f'{c.id}: {c.input_types} -> {c.return_type}' for c in bucket.candidates))
                    raise
        solved = solve_search_graph(graph,fixed_selections=fixed,dump_subdirectory='NativePrefillDiagnostic')
        points,records = distribution_selection_state(solved,policy=self.distribution_policy.identity)
        old = {p.id for p in module.selection_points if p.kind=='distribution'}
        proposed = fm.verify_module(replace(module,
            selection_points=tuple(p for p in module.selection_points if p.id not in old)+points,
            selections=tuple(r for r in module.selections if r.point_id not in old)+records,
            metadata={**module.metadata,'auto_distribution_proposal':{
                'schema':'flagmega.auto-distributed-proposal/v1','placement':graph.placement.to_data(),
                'solver':'ortools-cp-sat','status':solved.status,'objective':solved.objective},
                'local_distribution_override':{'policy':'glm53-native-bf16-mma-threaded-fma-full-k-w8/v3',
                    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'constraints':tuple(sorted(fixed.items()))}}))
        remember_proposal(proposed,graph,self)
        (trial/'distribution-solve.json').write_text(json.dumps({'fixed':fixed,'status':solved.status},indent=2)+'\n')
        return proposed


def native_implementation_model():
    base=Tp4DistributionTarget().triton_implementation_model
    dense=('tir.dense_matmul.packed_k_major_mma','tir.dense_matmul.packed_k_major_threaded_fma')
    families=('sparse_experts_gate_up','partitioned_sparse_experts_down')
    identities=(*dense,*(f'tir.{family}.wgmma' for family in families),'tir.mhc_pre.stream_fma','tir.mhc_pre.contiguous_tf32')
    def configure(implementation):
        parameters = dict(implementation.parameters)
        if implementation.id in identities:
            parameters['compute_num_warps'] = 8
        if implementation.id in {'tir.block_fp8.mma', 'tir.matmul_glu.mma'}:
            parameters['scaled_accumulation'] = 'fma'
        if implementation.id == 'tir.dsa_paged_attention.decode':
            parameters.update(decode_schedule='bf16_mma', compute_num_warps=8)
        if implementation.family == 'kda_gated_recurrent_core':
            parameters.update(reduction_layout='warp_k', compute_num_warps=8)
        if implementation.family == 'gated_norm_apply':
            parameters['inverse_std_algorithm'] = 'sqrt_div'
        return replace(implementation, parameters=parameters)
    catalog=replace(base,implementations=tuple(configure(i) for i in base.implementations),preferences={**base.preferences,'dense_matmul':(*dense,*(i for i in base.preferences['dense_matmul'] if i not in dense)),**{family:(f'tir.{family}.wgmma',*base.preferences.get(family,())) for family in families}})
    return catalog


def rewrite_attention_schedules(module):
    """Retain the native BF16 QK/PV accumulator order in resumed artifacts."""
    definitions = []
    for definition in module.kernel_definitions:
        dispatch = definition.dispatch
        microkernel = dispatch.microkernel
        if microkernel.family == 'dsa_paged_attention':
            assert dispatch.semantic_attrs['probability_rounding'] == 'online_bf16'
            assert microkernel.variant == 'decode'
            microkernel = replace(microkernel, parameters={**microkernel.parameters,
                                  'decode_schedule': 'bf16_mma', 'compute_num_warps': 8})
            definition = replace(definition, dispatch=replace(dispatch, microkernel=microkernel,
                semantic_attrs={**dispatch.semantic_attrs, 'index_traversal': 'compact'}))
        definitions.append(definition)
    return tuple(definitions)


def rewrite_gated_norm_schedules(module):
    """Retain the native gated RMSNorm's separate sqrt and reciprocal."""
    definitions = []
    for definition in module.kernel_definitions:
        dispatch = definition.dispatch
        microkernel = dispatch.microkernel
        if microkernel.family == 'gated_norm_apply':
            microkernel = replace(microkernel, parameters={**microkernel.parameters,
                                  'inverse_std_algorithm': 'sqrt_div'})
            definition = replace(definition, dispatch=replace(dispatch, microkernel=microkernel))
        definitions.append(definition)
    return tuple(definitions)


def rewrite_qkv_norm_schedules(module):
    """Match fused Q/KV RMS grouping while retaining the selected shard ABI."""
    by_name = {definition.name: definition for definition in module.kernel_definitions}
    node_map = module.node_map
    statistics, uses = set(), {}
    for node in module.nodes:
        if node.op != 'tir.call':
            continue
        name = str(node.attrs['callee'])
        if name not in by_name or by_name[name].dispatch.microkernel.family != 'norm_stats':
            continue
        source = fm.logical_type(node_map[node.inputs[0]].type)
        dtype = source.dtype.elem_type if isinstance(source.dtype, fm.VectorType) else source.dtype
        selected = dtype == fm.DType.BFLOAT16
        uses.setdefault(name, set()).add(selected)
        if selected:
            statistics.add(name)
    assert all(len(values) == 1 for values in uses.values()), 'Mixed precision statistics share one kernel'
    finalizers, uses = set(), {}
    for node in module.nodes:
        if node.op != 'tir.call':
            continue
        name = str(node.attrs['callee'])
        if name not in by_name or by_name[name].dispatch.microkernel.family != 'gather_reduce_norm_apply':
            continue
        producer = node_map[node.inputs[0]]
        while producer.op == 'tir.buffer_view':
            producer = node_map[producer.inputs[0]]
        selected = producer.op == 'tir.call' and str(producer.attrs['callee']) in statistics
        uses.setdefault(name, set()).add(selected)
        if selected:
            finalizers.add(name)
    assert all(len(values) == 1 for values in uses.values()), 'Different statistics orders share one finalizer'
    definitions = []
    for definition in module.kernel_definitions:
        parameters = None
        if definition.name in statistics:
            parameters = {'block_size': 2048, 'reduction_layout': 'contiguous_8'}
        elif definition.name in finalizers:
            parameters = {'partial_reduction': 'global_tree'}
        if parameters is not None:
            dispatch = definition.dispatch
            microkernel = replace(dispatch.microkernel,
                                  parameters={**dispatch.microkernel.parameters, **parameters})
            definition = replace(definition, dispatch=replace(dispatch, microkernel=microkernel))
        definitions.append(definition)
    return tuple(definitions)


def rewrite_fp8_scaled_accumulation(module):
    """Apply the measured FlashInfer small-M promotion to selected SM90 calls."""
    definitions = []
    for definition in module.kernel_definitions:
        dispatch = definition.dispatch
        microkernel = dispatch.microkernel
        if (microkernel.family in {'block_fp8', 'matmul_glu'}
                and dispatch.semantic_attrs.get('activation_quantization') == 'dynamic_fp8_sm90'):
            assert microkernel.variant == 'mma', (definition.name, microkernel.variant)
            microkernel = replace(microkernel, parameters={**microkernel.parameters, 'scaled_accumulation': 'fma'})
            definition = replace(definition, dispatch=replace(dispatch, microkernel=microkernel))
        definitions.append(definition)
    return tuple(definitions)


def rewrite_mla_schedules(module):
    expected = {
        **{f'decode_mla_attention_sparse_self_attn_o_proj_weight_apply.partition{p}.dot': 2
           for p in range(4)},
        'decode_mla_attention_sparse_self_attn_kv_a_proj_with_mqa_weight_apply.packed': 2,
        'decode_mla_attention_sparse_self_attn_q_a_proj_weight_apply.packed': 2,
    }
    calls = {str(n.id): n for n in module.nodes if n.op == 'tir.call' and str(n.id) in expected}
    assert set(calls) == set(expected), set(expected) - set(calls)
    schedules = {}
    for identity, node in calls.items():
        name = str(node.attrs['callee'])
        assert name not in schedules or schedules[name] == expected[identity]
        schedules[name] = expected[identity]
    users = {str(n.id) for n in module.nodes if n.op == 'tir.call' and str(n.attrs['callee']) in schedules}
    assert users == set(expected), users
    definitions = []
    for definition in module.kernel_definitions:
        if definition.name in schedules:
            dispatch = definition.dispatch
            assert dispatch.microkernel.family == 'block_fp8' and dispatch.microkernel.variant == 'mma'
            assert dispatch.semantic_attrs['activation_quantization'] == 'none'
            microkernel = replace(dispatch.microkernel, parameters={**dispatch.microkernel.parameters,
                                  'dequant_mma_splits': schedules[definition.name]})
            definition = replace(definition, dispatch=replace(dispatch, microkernel=microkernel))
        definitions.append(definition)
    print({'kernel_rewrite': 'cuBLAS-profiler-backed MLA Q_A/KV_A and o_proj split2; Q_B keeps its accumulator',
           'schedules': schedules, 'calls': expected}, flush=True)
    return tuple(definitions)
