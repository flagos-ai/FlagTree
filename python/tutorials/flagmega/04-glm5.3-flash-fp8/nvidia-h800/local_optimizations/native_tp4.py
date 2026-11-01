"""Workload-local complete-K and FP8 MoE choices for the explicit native profile."""
import hashlib
import json
from dataclasses import replace
from math import prod
from pathlib import Path

from local_optimizations.tp4_distribution import (
    OUTPUT_SPLITS,
    Tp4DistributionTarget,
    _moe_contracts,
    select_moe_layout,
    select_output_split,
    select_head_shards,
)
from triton.flagmega import ir as fm
from triton.flagmega.passes.auto_distributed import AutoDistributedPass, solve_search_graph
from triton.flagmega.passes.auto_distributed.materializer import distribution_selection_state
from triton.flagmega.passes.auto_distributed.proposal_analysis import remember_proposal
from local_optimizations import kda_distribution


def select_complete_projection(bucket, groups, placement):
    """Choose output owners without partitioning the logical K reduction."""
    def broadcast(value):
        return (not isinstance(value, fm.DistributedType) or value.partial is None
                and value.exclusive is None and all(p == fm.SBP.broadcast() for p in value.axis_policies))

    def packet_split(policy):
        cyclic = False
        for stage in policy.stages:
            levels = {placement.hierarchy_levels[axis] for axis in stage.hierarchy_axes}
            if levels == {'b'}:
                if stage.distribution != fm.BlockCyclicSplit(1):
                    return False
                cyclic = True
            elif 'b' in levels or not isinstance(stage.distribution, fm.ContiguousSplit):
                return False
        return cyclic

    axes_order = ((0, 1, 2), (1, 2), (0, 2), (0, 1), (2,), (1,), (0,))
    block_owners = prod(size for size, level in zip(
        placement.hierarchy, placement.hierarchy_levels, strict=True) if level == 'b')
    if groups < block_owners:
        # Small complete-K projections already fit on card-local owners.
        # A chip split adds publication without increasing useful output work.
        axes_order = tuple(sorted(axes_order, key=lambda axes: any(
            placement.hierarchy_levels[axis] != 'b' for axis in axes)))
    for axes in axes_order:
        if prod(placement.hierarchy[a] for a in axes) > groups:
            continue
        candidates = [candidate for candidate in bucket.candidates
            if isinstance(candidate.return_type, fm.DistributedType)
            and candidate.return_type.partial is None and candidate.return_type.exclusive is None
            and isinstance(candidate.return_type.axis_policies[-1], fm.SBPSplit)
            and tuple(candidate.return_type.axis_policies[-1].hierarchy_axes) == axes
            and broadcast(candidate.input_types[0])]
        contiguous = [candidate for candidate in candidates if all(
            isinstance(stage.distribution, fm.ContiguousSplit)
            for stage in candidate.return_type.axis_policies[-1].stages)]
        # A ragged output can still use every block owner. Requiring an exact
        # contiguous split discarded these legal packed layouts and moved the
        # projection (and its following norm statistics) across chips instead.
        cyclic = [candidate for candidate in candidates if packet_split(candidate.return_type.axis_policies[-1])]
        matches = contiguous or cyclic
        if matches:
            assert len(matches) == 1, (bucket.node_id, axes, [c.id for c in matches])
            return matches[0]

    matches = [c for c in bucket.candidates if broadcast(c.return_type)
               and all(broadcast(t) for t in c.input_types)]
    assert len(matches) == 1, (bucket.node_id, groups, [c.id for c in matches])
    return matches[0]


class NativeTp4DistributionTarget(Tp4DistributionTarget):
    def __init__(self, *args, **kwargs):
        explicit_bufferization = kwargs.get('bufferization_policy') is not None
        super().__init__(*args, **kwargs)
        if not explicit_bufferization:
            self.bufferization_policy = self.bufferization_policy.with_options(replace(
                self.bufferization_policy.options, rdata_chip_sharding=True,
                barrier_bytes_budget=65536, barrier_fixpoint_rounds=1))

    def propose_tir(self, module):
        from local_optimizations import block_fp8_cooperative_schedule, coarse_decode_schedule, glu_cooperative_schedule, materialized_contracts, moe_cooperative_schedule, router_register_schedule, small_mma_prefetch_schedule

        module = materialized_contracts.prepare(module)
        if module.metadata.get('offline_projection_weight_decode'):
            from triton.flagmega.passes.tir.fuse_packed_accumulator_cast import fuse_packed_accumulator_casts
            module = fuse_packed_accumulator_casts(module,
                predicate=lambda node: 'offline_projection_weight_decode' in node.metadata)
        proposal = super().propose_tir(module)
        if self.triton_implementation_model.implementation(small_mma_prefetch_schedule.IMPLEMENTATION) is not None:
            proposal = small_mma_prefetch_schedule.select(proposal)
        if self.triton_implementation_model.implementation(moe_cooperative_schedule.IMPLEMENTATION) is not None:
            proposal = moe_cooperative_schedule.select(proposal)
        if self.triton_implementation_model.implementation(block_fp8_cooperative_schedule.IMPLEMENTATION) is not None:
            proposal = block_fp8_cooperative_schedule.select(proposal)
        if self.triton_implementation_model.implementation(glu_cooperative_schedule.IMPLEMENTATION) is not None:
            proposal = glu_cooperative_schedule.select(proposal)
        if module.metadata.get('offline_projection_weight_decode'):
            from local_optimizations.offline_projection_weights import select
            proposal = select(proposal)
        if self.triton_implementation_model.implementation(router_register_schedule.IMPLEMENTATION) is not None:
            proposal = router_register_schedule.select(proposal)
        if self.triton_implementation_model.implementation(coarse_decode_schedule.COMPLETE_K) is not None:
            proposal = coarse_decode_schedule.select(proposal)
        return proposal

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
        for name,bucket in graph.bucket_map.items():
            node=module.node_map[name]
            if node.op!='ntt.packed_matmul':continue
            lhs=fm.logical_type(module.node_map[node.inputs[0]].type)
            if lhs.dtype is not fm.DType.BFLOAT16:continue
            groups=fm.logical_type(node.type).shape[-1].fixed_value
            fixed[name]=select_complete_projection(bucket, groups, graph.placement).id
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
        fixed.update(select_head_shards(graph))
        fixed.update(kda_distribution.selections(graph))
        from local_optimizations import partitioned_projection_distribution
        fixed.update(partitioned_projection_distribution.selections(graph))
        solved = solve_search_graph(graph,fixed_selections=fixed,dump_subdirectory='NativePrefillDiagnostic')
        points,records = distribution_selection_state(solved,policy=self.distribution_policy.identity)
        old = {p.id for p in module.selection_points if p.kind=='distribution'}
        proposed = fm.verify_module(replace(module,
            selection_points=tuple(p for p in module.selection_points if p.id not in old)+points,
            selections=tuple(r for r in module.selections if r.point_id not in old)+records,
            metadata={**module.metadata,'auto_distribution_proposal':{
                'schema':'flagmega.auto-distributed-proposal/v1','placement':graph.placement.to_data(),
                'solver':'ortools-cp-sat','status':solved.status,'objective':solved.objective},
                'local_distribution_override':{'policy':'glm53-native-full-k-kda-mla-chip-heads-w8/v5',
                    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    'head_policy':kda_distribution.POLICY,
                    'head_policy_sha256':hashlib.sha256(Path(kda_distribution.__file__).read_bytes()).hexdigest(),
                    'constraints':tuple(sorted(fixed.items()))}}))
        remember_proposal(proposed,graph,self)
        (trial/'distribution-solve.json').write_text(json.dumps({'fixed':fixed,'status':solved.status},indent=2)+'\n')
        return proposed


def native_implementation_model(*, offline_projection_weights=False):
    from local_optimizations import block_fp8_cooperative_schedule, coarse_decode_schedule, glu_cooperative_schedule, moe_cooperative_schedule, reduction_rows_schedule, remote_gather_schedule, remote_reduce_schedule, router_register_schedule, small_mma_prefetch_schedule
    from triton.flagmega.targets.nvidia.mhc import mhc_cooperative_stream_fma_implementations

    base=Tp4DistributionTarget().triton_implementation_model
    dense=('tir.dense_matmul.packed_k_major_mma','tir.dense_matmul.packed_k_major_threaded_fma')
    families=('sparse_experts_gate_up','partitioned_sparse_experts_down')
    identities=(*dense,*(f'tir.{family}.wgmma' for family in families),'tir.mhc_pre.stream_fma','tir.mhc_pre.contiguous_tf32')
    def configure(implementation):
        parameters = dict(implementation.parameters)
        facts = dict(implementation.facts)
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
        if implementation.id == 'tir.mla_projection.mma':
            # The local override assigns output tiles to different CTAs.
            facts['result_memory_scope'] = 'chip'
        if implementation.family in {'mhc_post', 'dsa_paged_attention'}:
            # Replica tasks write disjoint parts of one shared result.
            facts['result_memory_scope'] = 'chip'
        return replace(implementation, parameters=parameters, facts=facts)
    preferred = {family: tuple(dict.fromkeys((f'tir.{family}.wgmma', *base.preferences.get(family, ()))))
                 for family in families}
    catalog=replace(base,implementations=tuple(configure(i) for i in base.implementations),preferences={**base.preferences,'dense_matmul':(*dense,*(i for i in base.preferences['dense_matmul'] if i not in dense)),**preferred})
    cooperative_mhc = mhc_cooperative_stream_fma_implementations()
    catalog = replace(catalog, implementations=(*catalog.implementations, *cooperative_mhc),
                      preferences={**catalog.preferences,
                                   'mhc_pre': tuple(value.id for value in cooperative_mhc)})
    if offline_projection_weights:
        from local_optimizations.offline_projection_weights import implementation_model
        catalog = implementation_model(catalog)
    catalog = router_register_schedule.implementation_model(remote_reduce_schedule.implementation_model(reduction_rows_schedule.implementation_model(remote_gather_schedule.implementation_model(glu_cooperative_schedule.implementation_model(
        block_fp8_cooperative_schedule.implementation_model(moe_cooperative_schedule.implementation_model(
            small_mma_prefetch_schedule.implementation_model(catalog))))))))
    return coarse_decode_schedule.implementation_model(catalog)


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
    """Match native RMS orders without changing the selected shard ABI."""
    by_name = {definition.name: definition for definition in module.kernel_definitions}
    node_map = module.node_map
    statistics, complete_statistics, uses = set(), set(), {}
    for node in module.nodes:
        if node.op != 'tir.call':
            continue
        name = str(node.attrs['callee'])
        if name not in by_name or by_name[name].dispatch.microkernel.family != 'norm_stats':
            continue
        source_type = node_map[node.inputs[0]].type
        source = fm.logical_type(source_type)
        dtype = source.dtype.elem_type if isinstance(source.dtype, fm.VectorType) else source.dtype
        selected = dtype == fm.DType.BFLOAT16
        dispatch = by_name[name].dispatch
        axis = int(dispatch.semantic_attrs.get('axis', -1)) % source.rank
        complete = dtype == fm.DType.FLOAT32 and not dispatch.semantic_attrs.get('use_mean', False) and (
            not isinstance(source_type, fm.DistributedType) or
            source_type.partial is None and source_type.exclusive is None and
            all(policy == fm.SBP.broadcast() for policy in source_type.axis_policies[axis:]))
        uses.setdefault(name, set()).add('bf16' if selected else 'complete_fp32' if complete else 'other')
        if selected:
            statistics.add(name)
        elif complete:
            complete_statistics.add(name)
    assert all(len(values) == 1 for values in uses.values()), 'Different RMS orders share one statistics kernel'
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
        elif definition.name in complete_statistics:
            parameters = {'reduction_algorithm': 'fma_1024'}
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
    if getattr(module, 'metadata', {}).get('offline_projection_weight_decode'):
        from local_optimizations.offline_projection_weights import verify_native_schedules
        return verify_native_schedules(module)
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
