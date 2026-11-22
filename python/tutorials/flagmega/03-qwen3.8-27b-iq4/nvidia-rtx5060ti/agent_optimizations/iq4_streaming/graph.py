"""Own the IQ4 distribution, selection and cross-entry memory contracts.

No packing or numerical representation is changed here. Physical row counts
are separate from packed tensor elements and the cooperative data mesh.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod
from collections.abc import Mapping
from copy import copy

from triton.flagmega.codegen.triton.candidates import (
    TritonCandidateProposal, TritonCandidateProviderRegistry,
)
from triton.flagmega.codegen.triton.implementation import TritonImplementationModel
from triton.flagmega.codegen.triton.selection import TritonTirSelectionPolicy
from triton.flagmega.errors import CodegenError, IRVerificationError
from triton.flagmega.ir import DistributedType, TensorType, TupleType, VectorType
from triton.flagmega.ir.distributed_inference import broadcast_ir_type, tensor_of
from triton.flagmega.ir.execution_mesh import ExecutionMesh, effective_execution_mesh
from triton.flagmega.ir.launch_recipe import TaskLaunchRecipe
from triton.flagmega.passes.auto_distributed.candidates import (
    DistributedCandidate, DistributedCandidateProviderBase, DistributedCandidateProviderRegistry,
)
from triton.flagmega.passes.tir.bufferize.policy import NttBufferizationPolicy
from triton.flagmega.targets.nvidia_sm120 import NvidiaSm120Target

PLAN_HASH = "7b04b547f07fd1c71e4ecb8242720a631470c2a49953ab115487f3a1f5a6c361"
RESIDENCY_ENCODING = {"size_per_thread": [1, 1], "threads_per_warp": [1, 32],
                      "warps_per_cta": [4, 1], "order": [1, 0]}
OPS = frozenset({"math.block_scaled_matmul", "math.packed_block_scaled_matmul",
                 "nn.matmul_glu", "nn.packed_matmul_glu", "nn.interleaved_projection"})
FAMILIES = {"math.block_scaled_matmul": "iq4_streaming_projection",
            "math.packed_block_scaled_matmul": "iq4_streaming_projection",
            "nn.matmul_glu": "iq4_streaming_glu", "nn.packed_matmul_glu": "iq4_streaming_glu",
            "nn.interleaved_projection": "iq4_streaming_interleaved"}


@dataclass(frozen=True)
class StreamingConfig:
    row_tile: int = 4
    kv_row_tile: int = 4

    def __post_init__(self):
        if self.row_tile != 4 or self.kv_row_tile != 4:
            raise ValueError("r001 v2.1 fixes R4 for every projection")

    def to_data(self):
        return {"schema": "flagmega.iq4-streaming-graph/r001-v2.1", "plan_hash": PLAN_HASH,
                "row_tile": self.row_tile, "kv_row_tile": self.kv_row_tile,
                "num_warps": 4, "word_bytes": 4, "program_kind": "ordinary",
                "residency_encoding": RESIDENCY_ENCODING,
                "row_encoding": {"sliced_dimension": 1, "parent": RESIDENCY_ENCODING},
                "shuffle": {"opcode": "shfl.sync.idx.b32", "membermask": 0xffffffff,
                            "clamp": 0x1f, "participants": 32}}


def is_iq4(node):
    if node.op not in OPS:
        return False
    if node.op in {"nn.matmul_glu", "nn.packed_matmul_glu"}:
        return all(node.attrs.get(k) == "IQ4_XS" for k in ("gate_weight_encoding", "up_weight_encoding"))
    return node.attrs.get("weight_encoding") == "IQ4_XS"


def scalar_rows(node):
    if node.op == "nn.interleaved_projection":
        # The physical weight has both Q/gate fields. The tuple contains
        # two packed fields; neither the packet count nor field count is N.
        if not isinstance(node.type, TupleType) or len(node.type.fields) != 2:
            raise CodegenError("IQ4 Q projection requires two result fields")
        return sum(_scalar_elements(field) for field in node.type.fields)
    output = tensor_of(node.type)
    if output.rank != 2 or output.shape[0].fixed_value != 1:
        raise CodegenError("r001 IQ4 streaming is restricted to B1 projections")
    return _scalar_elements(node.type)


def _scalar_elements(value):
    tensor = tensor_of(value)
    if any(not d.is_fixed for d in tensor.shape):
        raise CodegenError("IQ4 row tasks require statically bounded output")
    lanes = tensor.dtype.lane_count if isinstance(tensor.dtype, VectorType) else 1
    return prod(d.fixed_value for d in tensor.shape) * lanes


def admission_compile_options(node):
    """Expert-adjudicated H001 caps, attached before program formation.

    These caps bind the normal runtime's --minnctapersm=1 specialization.
    Q requires five CTAs; both GLUs require four CTAs and at most 107 regs.
    """
    family = FAMILIES[node.op]
    if node.id == "decode_attention_1_q.packed":
        return {"kernel_maxnreg": 96}
    if node.id == "decode_attention_1_mlp_gate_up":
        return {"kernel_maxnreg": 104}
    if family == "iq4_streaming_projection":
        return {"kernel_maxnreg": 80}
    if family == "iq4_streaming_glu" and node.id == "decode_linear_0_mlp_gate_up":
        return {"kernel_maxnreg": 104}
    return {}


def prepare_distribution(module, config):
    """Run before proposal on every fresh/resumed public compilation."""
    previous = module.metadata.get("iq4_streaming")
    if previous is not None and canonical_data(previous) != canonical_data(config.to_data()):
        raise IRVerificationError("Changing IQ4 native identity requires resuming before distribution", stage=module.stage)
    if previous is not None and canonical_data(previous) == canonical_data(config.to_data()):
        return module
    if module.stage not in {"unused_functions_removed", "distribution_constants_frozen"}:
        raise IRVerificationError("Install IQ4 scheduling at complete AutoPacking/After", stage=module.stage)
    rewritten = tuple(replace(node, metadata={**node.metadata,
        **admission_compile_options(node),
        "execution_mesh": ExecutionMesh.dynamic().to_data(), "kernel_program_kind": "ordinary",
        "kernel_num_warps": 4, "iq4_streaming_plan_hash": PLAN_HASH,
        "iq4_streaming_residency_encoding": RESIDENCY_ENCODING})
        if is_iq4(node) else node for node in module.nodes)
    return replace(module, nodes=rewritten, metadata={**module.metadata, "iq4_streaming": config.to_data()})


class IndependentRowsProvider(DistributedCandidateProviderBase):
    allows_partial_inputs = False
    is_exhaustive = True

    def __init__(self, op):
        self.op_names = frozenset({op})

    def _enumerate_candidates(self, context):
        node = context.source_call
        if not is_iq4(node):
            raise IRVerificationError("Streaming provider received a non-IQ4 operation", node_id=node.id)
        if context.placement.physical_level_size("b") != 1:
            raise IRVerificationError("IQ4 row work must leave the cooperative owner mesh", node_id=node.id)
        return (DistributedCandidate("independent-rows", broadcast_ir_type(node.type, context.placement),
            tuple(broadcast_ir_type(context.module.node_map[i].type, context.placement) for i in node.inputs),
            0, "exclusive-full-K-row-tasks", node.op, "agent", "flagmega.iq4-streaming/r001-v2.1", (PLAN_HASH,)),)


class ConditionalDistributionProvider:
    """Preserve all unrelated provider inverse/forward inference methods."""
    def __init__(self, op, fallback):
        self.op_names = frozenset({op})
        self.fallback = fallback
        self.streaming = IndependentRowsProvider(op)
        self.is_exhaustive = fallback.is_exhaustive
        self.allows_partial_inputs = fallback.allows_partial_inputs

    def __getattr__(self, name):
        def delegate(context, *args, **kwargs):
            provider = self.streaming if is_iq4(context.source_call) else self.fallback
            return getattr(provider, name)(context, *args, **kwargs)
        return delegate


class StreamingDistributionPolicy:
    def __init__(self, base, config):
        self.base, self.config = base, config

    @property
    def identity(self):
        return self.base.identity + "+iq4-exclusive-rows-r001-v2.1:" + PLAN_HASH

    def __getattr__(self, name):
        return getattr(self.base, name)

    def with_placement(self, placement):
        # Preserve the installed split/reshard/provider policy when rebinding
        # topology. Reconstructing the default loses explicit policy options.
        if hasattr(self.base, "with_placement"):
            base = self.base.with_placement(placement)
        elif hasattr(self.base, "_placements"):
            base = copy(self.base)
            base._placements = (placement,)
        else:
            raise CodegenError("Installed distribution policy cannot preserve a placement rebind")
        return StreamingDistributionPolicy(base, self.config)

    def register_candidate_providers(self, registry):
        original = DistributedCandidateProviderRegistry()
        self.base.register_candidate_providers(original)
        registry.set_split_candidate_provider(original.split_candidate_provider)
        for op in original.op_names:
            provider = original.try_get(op)
            registry.add(ConditionalDistributionProvider(op, provider) if op in OPS
                         else _SingleOpDistributionProvider(op, provider))


class _SingleOpDistributionProvider:
    def __init__(self, op, base):
        self.op_names, self.base = frozenset({op}), base

    def __getattr__(self, name):
        return getattr(self.base, name)


class StreamingTIRProvider:
    def __init__(self, base, config):
        self.op_names, self.base, self.config = base.op_names, base, config

    def propose(self, node, context):
        if not is_iq4(node):
            return self.base.propose(node, context)
        mesh = effective_execution_mesh(context.module, node)
        if mesh != ExecutionMesh.dynamic():
            raise CodegenError("Selected IQ4 source has stale cooperative distribution")
        if node.metadata.get("iq4_streaming_plan_hash") != PLAN_HASH:
            raise CodegenError("Selected IQ4 source has stale native identity; resume full packing")
        rows = scalar_rows(node)
        tile = self.config.kv_row_tile if rows == 1024 else self.config.row_tile
        family = FAMILIES[node.op]
        implementations = tuple(i for i in context.implementations(family)
                                if native_contract_matches(i, tile))
        if not implementations:
            raise CodegenError(f"Missing qualified native implementation for {family} R{tile}; no fallback")
        recipe = TaskLaunchRecipe(rows, tile, "ordinary").to_data()
        candidates = tuple(context.configure_implementation(i, semantic_parameters={"task_mapping": recipe},
            facts={"source_memory_scope": "chip", "result_memory_scope": "chip", "required_codegen_backend": "triton",
                   "iq4_streaming_plan_hash": PLAN_HASH})
                           for i in implementations)
        return TritonCandidateProposal(candidates, context.choose_default(family, candidates))


def native_contract_matches(implementation, tile=4):
    p = implementation.parameters
    return (tile == 4 and p.get("tile_n") == 4 and p.get("num_warps") == 4
            and p.get("plan_hash") == PLAN_HASH
            and canonical_data(p.get("residency_encoding")) == canonical_data(RESIDENCY_ENCODING)
            and implementation.facts.get("required_codegen_backend") == "triton"
            and implementation.facts.get("internal_grid_barriers") == 0)


def canonical_data(value):
    if isinstance(value, Mapping):
        return {str(k): canonical_data(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(canonical_data(v) for v in value)
    return value


_VIEWS = frozenset({"builtin.get_item", "builtin.tuple", "builtin.identity", "tir.buffer_view",
                    "tir.buffer_subspan", "distributed.sharded_view", "tensors.bitcast", "tensors.reshape"})


class StreamingBufferizationPolicy(NttBufferizationPolicy):
    """Keep task inputs/outputs and structural aliases in the CHIP pool."""
    def plan_function_memory(self, module):
        calls = {n.id for n in module.nodes if (n.metadata.get("iq4_streaming_plan_hash") == PLAN_HASH or n.metadata.get("iq4_q8_plan_hash") == Q8_PLAN)}
        selected = set(calls)
        nodes = module.node_map
        # Distribution may introduce an additional fixed-mesh program cut
        # beside a boxing/norm boundary. Those edges need the same lifetime
        # proof as the dynamic projection edges.
        owners = _program_call_owners(module)
        for consumer in module.nodes:
            if consumer.id not in owners:
                continue
            for input_id in consumer.inputs:
                pending = [(input_id, ())]
                visited = set()
                while pending:
                    value, path = pending.pop()
                    if value in visited:
                        continue
                    visited.add(value)
                    path = (*path, value)
                    if value in owners and owners[value] != owners[consumer.id]:
                        selected.update(path)
                    elif nodes[value].op in _VIEWS:
                        pending.extend((i, path) for i in nodes[value].inputs)
        for call in calls:
            pending = list(nodes[call].inputs)
            visited = set()
            while pending:
                value = pending.pop()
                if value in visited:
                    continue
                visited.add(value)
                selected.add(value)
                if nodes[value].op in _VIEWS:
                    pending.extend(nodes[value].inputs)
        # Alias leaves cannot be given a private per-CTA offset either.
        changed = True
        while changed:
            additions = {n.id for n in module.nodes if n.op in _VIEWS and set(n.inputs) & selected}
            changed = not additions.issubset(selected)
            selected.update(additions)
        # Weight-bank formals, external state and aliases of readonly input
        # storage already have canonical CHIP addresses. A function-pool
        # placement annotation would incorrectly contradict their ABI.
        from triton.flagmega.ir import kernel_dispatch_for_call

        def tensor_leaves(value):
            return (isinstance(value, (TensorType, DistributedType)) or
                    isinstance(value, TupleType) and all(tensor_leaves(f) for f in value.fields))

        def has_temporary_root(value, visited=None):
            visited = set() if visited is None else visited
            if value in visited:
                return False
            visited.add(value)
            node = nodes[value]
            if kernel_dispatch_for_call(module, node) is not None:
                return True
            return node.op in _VIEWS and any(has_temporary_root(i, visited) for i in node.inputs)

        external_aliases = {i for f in module.functions for i in (*f.parameters, *f.outputs)}
        changed = True
        while changed:
            additions = {i for n in module.nodes if n.op in _VIEWS
                         and (n.id in external_aliases or set(n.inputs) & external_aliases)
                         for i in (n.id, *n.inputs)}
            changed = not additions.issubset(external_aliases)
            external_aliases.update(additions)
        selected = {i for i in selected if i not in external_aliases
                    and tensor_leaves(nodes[i].type) and has_temporary_root(i)}
        module = replace(module, nodes=tuple(replace(n, metadata={**n.metadata,
            "bufferization.memory_space": self.options.workspace}) if n.id in selected else n
            for n in module.nodes), metadata={**module.metadata,
            "iq4_streaming_chip_edges": sorted(selected)})
        return super().plan_function_memory(module)


def _program_call_owners(module):
    from triton.flagmega.ir import kernel_dispatch_for_call
    from triton.flagmega.passes.functions.graph import function_nodes
    owners = {}

    def visit(node, symbol):
        if kernel_dispatch_for_call(module, node) is not None:
            owners[node.id] = symbol
            return
        callee = module.function_map.get(str(node.attrs.get("callee", "")))
        if callee is not None:
            for child in function_nodes(module, callee):
                if child.op in {"tir.call", "control.for"}:
                    visit(child, symbol)

    for program in module.metadata.get("kernel_program_partition", ()):
        for call in program["calls"]:
            node = module.node_map.get(call)
            if node is None:
                node = next((n for n in module.nodes if n.op == "control.for"
                             and call.endswith("." + n.id + "_iv")), None)
            if node is not None:
                visit(node, program["symbol"])
    return owners


class StreamingTarget(NvidiaSm120Target):
    def auto_distribute(self, module):
        return super().auto_distribute(prepare_distribution(module, self.streaming_config))

    def propose_distribution(self, module):
        return super().propose_distribution(prepare_distribution(module, self.streaming_config))

    def with_placement(self, placement):
        base = NvidiaSm120Target.with_placement(self, placement)
        # Reinstall the distribution wrapper after reconstruction. Other
        # policies, model, options and invocation backend must survive.
        base.streaming_config = self.streaming_config
        base.distribution_policy = self.distribution_policy.with_placement(placement)
        base.codegen_backend = self.codegen_backend
        return base


def configure_target(target, config=StreamingConfig(), *, native_model=None):
    if target.name != "nvidia-sm120" or getattr(target, "codegen_backend", "triton") != "triton":
        raise CodegenError("r001 v2.1 requires the SM120 Triton target")
    if isinstance(target, StreamingTarget):
        if target.streaming_config != config:
            raise CodegenError("Target already installed with different row tiles")
        installed = tuple(i for i in target.triton_implementation_model.implementations
                          if i.family in set(FAMILIES.values()))
        if installed:
            if not all(native_contract_matches(i) for i in installed):
                raise CodegenError("Target has stale native implementations; rebuild from full packing")
            return target
    if native_model is None:
        try:
            from . import native
        except ImportError as error:
            raise CodegenError("Kernel optimizer native delivery is pending: iq4_streaming/native.py; no fallback") from error
        native.register()
        native_model = native.implementations(row_tile=config.row_tile, kv_row_tile=config.kv_row_tile)
    # Candidate IR deeply freezes metadata. Freeze catalog sequences as well
    # so the target verifier compares identical nested parameter semantics.
    native_model = tuple(replace(i, parameters=canonical_data(i.parameters)) for i in native_model)
    if (set(i.family for i in native_model) != set(FAMILIES.values())
            or not all(native_contract_matches(i) for i in native_model)):
        raise CodegenError("r001 E32 native delivery is pending or incompatible; no fallback")
    old = target.triton_implementation_model
    model = TritonImplementationModel((*old.implementations, *native_model),
        {**old.preferences, **{family: tuple(i.id for i in native_model if i.family == family)
                             for family in set(FAMILIES.values())}}, name=old.name + "+iq4-streaming-r001-v2.1:" + PLAN_HASH)
    registry = TritonCandidateProviderRegistry()
    for provider in target.tir_selection_policy.registry.providers:
        registry.add(StreamingTIRProvider(provider, config) if provider.op_names & OPS else provider)
    configured = target if isinstance(target, StreamingTarget) else configure_distribution_target(target, config)
    configured.streaming_config = config
    configured.tir_selection_policy = TritonTirSelectionPolicy(target.tir_selection_policy.workspace_annotator, registry)
    configured.triton_implementation_model = model
    configured.bufferization_policy = StreamingBufferizationPolicy(target.bufferization_policy.options,
        synchronization_protocol_provider=target.bufferization_policy.synchronization_protocol_provider)
    configured.codegen_backend = "triton"
    return configured


def configure_distribution_target(target, config=StreamingConfig()):
    """Graph-only preparation can be tested before native integration arrives.

    The caller must stop before TIR selection in this mode; this does not
    represent a runnable candidate or permit the old SIMT path to execute.
    """
    if target.name != "nvidia-sm120" or getattr(target, "codegen_backend", "triton") != "triton":
        raise CodegenError("r001 v2.1 requires SM120 Triton")
    configured = StreamingTarget(target.machine.capability, options=target.options,
        vectorization_policy=target.vectorization_policy,
        distribution_policy=StreamingDistributionPolicy(target.distribution_policy, config),
        selection_policy=target.selection_policy,
        bufferization_policy=StreamingBufferizationPolicy(target.bufferization_policy.options,
            synchronization_protocol_provider=target.bufferization_policy.synchronization_protocol_provider),
        packing_policy=target.packing_policy, tir_selection_policy=target.tir_selection_policy,
        tir_lowering_policy=target.tir_lowering_policy,
        microkernel_selection_policy=target.microkernel_selection_policy,
        triton_implementation_model=target.triton_implementation_model)
    configured.streaming_config = config
    configured.codegen_backend = getattr(target, "codegen_backend", "triton")
    return configured

# r003 is an independent semantic/candidate path; old R4 identities stay old.
from triton.flagmega.ir.ops.math._iq4_q8 import (
    SCHEMA as Q8_SCHEMA, PLAN as Q8_PLAN, NUM as Q8_NUM, INTERFACE as Q8_INTERFACE,
    ABI as Q8_ABI, OPS as Q8_FAMILIES,
)

@dataclass(frozen=True)
class Q8Config:
    row_tile: int = 1
    num_warps: int = 4
    def __post_init__(self):
        if (self.row_tile,self.num_warps)!=(1,4): raise CodegenError('r003 fixes row1/K4L32.')
    def to_data(self):
        return dict(schema=Q8_SCHEMA,plan_hash=Q8_PLAN,numerical_profile='H003-N-v2',
                    numerical_profile_hash=Q8_NUM,native_interface=Q8_INTERFACE,q8_abi=Q8_ABI,
                    row_tile=1,num_warps=4,threads=128,sharing='unshared-337',launch_credit=0)

def q8_node_options(node):
    family=Q8_FAMILIES[node.op]
    quant=family=='q8_1_prepare'; glu=family=='iq4_q8_glu'; inter=family=='iq4_q8_interleaved'
    return dict(kernel_maxnreg=104 if glu else 96 if inter else 80,kernel_num_warps=4,
                kernel_enable_static_shared=not quant,kernel_program_kind='ordinary',
                iq4_q8_plan_hash=Q8_PLAN,iq4_q8_native_interface=Q8_INTERFACE,
                iq4_q8_dynamic_shared_bytes=0,iq4_q8_logical_static_bytes=0 if quant else 768 if glu else 384,
                iq4_q8_min_cta_per_sm=4 if glu else 5 if inter else 6)

def verify_q8_nodes(module,*,configured=True):
    """Unconditional node and origin ledger check, including same-config calls."""
    from triton.flagmega.ir import get_definition
    ledger=[]; nodes=module.node_map
    for n in module.nodes:
        if n.op not in Q8_FAMILIES: continue
        get_definition(n.op).verify(n,module)
        a=n.attrs
        identity=dict(graph_schema=Q8_SCHEMA,plan_hash=Q8_PLAN,numerical_profile='H003-N-v2',
                      numerical_profile_hash=Q8_NUM,native_interface=Q8_INTERFACE,q8_abi=Q8_ABI)
        if any(a.get(k)!=v for k,v in identity.items()): raise CodegenError(f'Q8 semantic identity lost at {n.id}.')
        if not n.metadata.get('iq4_q8_origin_id') or not n.metadata.get('iq4_q8_ready_publication'):
            raise CodegenError(f'Q8 origin/ready publication lost at {n.id}.')
        if n.op!='math.q8_1_prepare':
            family=Q8_FAMILIES[n.op]
            allowed={(10240,5120),(6144,5120),(5120,6144),(5120,17408),(1024,5120),(248320,5120)} if family=='iq4_q8_projection' else {(17408,5120)} if family=='iq4_q8_glu' else {(12288,5120)}
            if (a['logical_n'],a['logical_k']) not in allowed:
                raise CodegenError(f'Q8 shape outside frozen full-model census: {n.id}.')
            if (a['output_data_type']=='float32') != (a['logical_n']==248320):
                raise CodegenError(f'Q8 dtype does not preserve head/ordinary ownership: {n.id}.')
            prep=nodes[n.inputs[0]]
            if prep.op!='math.q8_1_prepare' or any(prep.attrs[k]!=a[k] for k in identity.keys() | {'logical_k','padded_k','group_count','source_dtype','source_version_key'}):
                raise CodegenError(f'Q8 consumer is not bound to its prepared version: {n.id}.')
        if configured and any(n.metadata.get(k)!=v for k,v in q8_node_options(n).items()):
            raise CodegenError(f'Q8 per-node options missing/stale: {n.id}.')
        if configured and effective_execution_mesh(module,n)!=ExecutionMesh.dynamic():
            raise CodegenError(f'Q8 must use ordinary independent tasks: {n.id}.')
        ledger.append(dict(id=n.id,op=n.op,inputs=n.inputs,attrs=canonical_data(a),
                           metadata=canonical_data(n.metadata),type=str(n.type)))
    if not ledger: raise CodegenError('No Q8 nodes in qualifying model.')
    return ledger

def prepare_q8_distribution(module,config=Q8Config()):
    if 'iq4_streaming' in module.metadata: raise CodegenError('Fresh r003 rejects old IQ4 marker.')
    previous=module.metadata.get('iq4_q8_r003')
    if previous is not None:
        if canonical_data(previous)!=canonical_data(config.to_data()): raise CodegenError('Changed Q8 identity requires a new unconfigured attempt.')
        verify_q8_nodes(module)
        return module
    if module.stage not in ('unused_functions_removed','distribution_constants_frozen'):
        raise CodegenError('Install r003 at fresh complete unused_functions_removed.')
    verify_q8_nodes(module,configured=False)
    result=replace(module,nodes=tuple(replace(n,metadata={**n.metadata,**q8_node_options(n),
        'execution_mesh':ExecutionMesh.dynamic().to_data()}) if n.op in Q8_FAMILIES else n for n in module.nodes),
        metadata={**module.metadata,'iq4_q8_r003':config.to_data()})
    verify_q8_nodes(result)
    return result

class Q8RowsProvider(IndependentRowsProvider):
    def _enumerate_candidates(self,context):
        n=context.source_call
        if n.op not in Q8_FAMILIES or context.placement.physical_level_size('b')!=1:
            raise CodegenError('Q8 only admits complete single-chip owner.')
        return (DistributedCandidate('q8-full-owner',broadcast_ir_type(n.type,context.placement),
            tuple(broadcast_ir_type(context.module.node_map[i].type,context.placement) for i in n.inputs),
            0,'full-K-independent-q8-tasks',n.op,'agent',Q8_SCHEMA,(Q8_PLAN,Q8_NUM)),)

class Q8DistributionPolicy(StreamingDistributionPolicy):
    @property
    def identity(self): return self.base.identity+'+q8-r003:'+Q8_PLAN
    def with_placement(self,placement):
        rebound=super().with_placement(placement)
        return Q8DistributionPolicy(rebound.base,self.config)
    def register_candidate_providers(self,registry):
        self.base.register_candidate_providers(registry)
        for op in Q8_FAMILIES: registry.add(Q8RowsProvider(op))

class Q8TIRProvider:
    op_names=frozenset(Q8_FAMILIES)
    def propose(self,node,context):
        verify_q8_nodes(context.module)
        family=Q8_FAMILIES[node.op]
        candidates=[]
        for i in context.implementations(family):
            if not q8_native_contract_matches(i): raise CodegenError('Stale Q8 native catalog.')
            quant=family=='q8_1_prepare'
            work=node.attrs['group_count'] if quant else node.attrs['logical_n']
            recipe=TaskLaunchRecipe(work,4 if quant else 1,'ordinary').to_data()
            candidates.append(context.configure_implementation(i,semantic_parameters={'task_mapping':recipe},
                facts={'source_memory_scope':'chip','result_memory_scope':'chip','required_codegen_backend':'triton',
                       'iq4_q8_plan_hash':Q8_PLAN,'iq4_q8_numerical_hash':Q8_NUM}))
        if not candidates: raise CodegenError(f'Missing Q8 native implementation: {family}.')
        return TritonCandidateProposal(tuple(candidates),context.choose_default(family,tuple(candidates)))

def q8_native_contract_matches(i):
    from triton.flagmega.codegen.triton.iq4_q8_renderer import implementation_parameters
    expected=implementation_parameters(i.family)
    return (i.family in Q8_FAMILIES.values() and i.variant=='q8_r003_v2'
            and canonical_data(i.parameters)==canonical_data(expected)
            and i.facts.get('required_codegen_backend')=='triton' and i.facts.get('internal_grid_barriers')==0)

class Q8Target(NvidiaSm120Target):
    def auto_distribute(self,module): return super().auto_distribute(prepare_q8_distribution(module,self.q8_config))
    def propose_distribution(self,module): return super().propose_distribution(prepare_q8_distribution(module,self.q8_config))
    def with_placement(self,placement):
        target=NvidiaSm120Target.with_placement(self,placement)
        return configure_q8_target(target,self.q8_config)

def configure_q8_target(target,config=Q8Config()):
    """Catalog owns native semantics. Verify it even on installed early return."""
    if target.name!='nvidia-sm120' or getattr(target,'codegen_backend','triton')!='triton': raise CodegenError('Q8 requires SM120 Triton.')
    if isinstance(target,StreamingTarget): raise CodegenError('Q8 cannot resume R4 configured target.')
    from .native import q8_implementations
    from triton.flagmega.codegen.triton.iq4_q8_renderer import register
    catalog=tuple(replace(i,parameters=canonical_data(i.parameters)) for i in q8_implementations(row_tile=1,num_warps=4))
    if set(i.family for i in catalog)!=set(Q8_FAMILIES.values()) or not all(q8_native_contract_matches(i) for i in catalog):
        raise CodegenError('Q8 catalog incomplete or stale.')
    register()
    if isinstance(target,Q8Target):
        installed=tuple(i for i in target.triton_implementation_model.implementations if i.family in Q8_FAMILIES.values())
        if target.q8_config!=config or len(installed)!=4 or not all(q8_native_contract_matches(i) for i in installed):
            raise CodegenError('Installed Q8 target is stale.')
        return target
    old=target.triton_implementation_model
    if any(i.family in set(FAMILIES.values()) | set(Q8_FAMILIES.values()) for i in old.implementations):
        raise CodegenError('Q8 target needs a fresh native model.')
    model=TritonImplementationModel((*old.implementations,*catalog),
        {**old.preferences,**{i.family:(i.id,) for i in catalog}},name=old.name+'+q8-r003:'+Q8_PLAN)
    registry=TritonCandidateProviderRegistry()
    for provider in target.tir_selection_policy.registry.providers: registry.add(provider)
    registry.add(Q8TIRProvider())
    configured=Q8Target(target.machine.capability,options=target.options,
        vectorization_policy=target.vectorization_policy,packing_policy=target.packing_policy,
        distribution_policy=Q8DistributionPolicy(target.distribution_policy,config),selection_policy=target.selection_policy,
        bufferization_policy=StreamingBufferizationPolicy(target.bufferization_policy.options,
            synchronization_protocol_provider=target.bufferization_policy.synchronization_protocol_provider),
        tir_selection_policy=TritonTirSelectionPolicy(target.tir_selection_policy.workspace_annotator,registry),
        tir_lowering_policy=target.tir_lowering_policy,microkernel_selection_policy=target.microkernel_selection_policy,
        triton_implementation_model=model)
    configured.q8_config=config; configured.enable_iq4_q8_preparation=True;configured.codegen_backend='triton'
    return configured

_configure_r4_target=configure_target

def configure_target(target,config=None,*,native_model=None):
    if config is None:
        config = Q8Config() if isinstance(target,Q8Target) else StreamingConfig()
    if isinstance(config,Q8Config):
        if native_model is not None: raise CodegenError('Q8 implementation model must come from kernel catalog.')
        return configure_q8_target(target,config)
    return _configure_r4_target(target,config,native_model=native_model)
