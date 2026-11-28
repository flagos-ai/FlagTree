# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""GU-only semantic rewrite and scoped distribution providers."""
from dataclasses import replace

from triton.flagmega.errors import IRSchemaError
from triton.flagmega.ir import DistributedType, Node, TupleType, get_definition, tensor_type, verify_module, ExecutionMesh
from triton.flagmega.ir.distributed_inference import broadcast_ir_type
from triton.flagmega.ir.ops.ntt.split_k_dense_glu import SplitKDenseGluProjection, SplitKDenseGluEpilogue
from triton.flagmega.passes.auto_distributed.candidates import DistributedCandidate, DistributedCandidateProviderBase


class _SplitGluDistributionProvider(DistributedCandidateProviderBase):
    op_names = frozenset((SplitKDenseGluProjection.op_name, SplitKDenseGluEpilogue.op_name))
    allows_partial_inputs = False
    is_exhaustive = True

    def _enumerate_candidates(self, context):
        node, placement = context.source_call, context.placement
        required = (48, 2) if node.op == SplitKDenseGluProjection.op_name else (24, )
        if placement.hierarchy != required or placement.hierarchy_levels != 'b' * len(required):
            return ()
        inputs = tuple(broadcast_ir_type(context.module.node_map[v].type, placement) for v in node.inputs)
        operands = tuple(replace(context.module.node_map[v], type=t) for v, t in zip(node.inputs, inputs, strict=True))
        definition = SplitKDenseGluProjection if node.op == SplitKDenseGluProjection.op_name else SplitKDenseGluEpilogue
        output = definition.infer_type(operands, node.attrs)
        return (DistributedCandidate('distribution.' + node.id + '.split_glu_physical_domain', output, inputs, 0,
                                     'explicit-complete-disjoint-GU-domain', objective_kind='agent',
                                     objective_model='flagmega.split-glu-domain/v1',
                                     objective_evidence=('48x2-split-projection-or-24x256-complete-epilogue', )), )


class SplitKDenseGluDistributionPolicy:
    """Add only the two explicit ops; preserve the supplied distribution policy."""

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):
        return getattr(self.base, name)

    @property
    def identity(self):
        return self.base.identity + '+split-k-dense-glu/v1'

    def register_candidate_providers(self, registry):
        self.base.register_candidate_providers(registry)
        registry.add(_SplitGluDistributionProvider())


def split_k_dense_glu_policy(target):
    if target.codegen_platform != 'nvidia' or target.codegen_architecture != 'sm90':
        raise IRSchemaError('The participant-retirement GU island requires the scoped SM90 implementation.')
    return SplitKDenseGluDistributionPolicy(target.distribution_policy)


def rewrite_split_k_dense_glu(module, *, node_id='decode_layer_mlp_gate_up'):
    """Replace one packed GU, lifting its paired readonly weight to every call.

    The input is an unallocated, packed semantic graph before distribution.
    All weights are recovered from the existing caller's actual builtin.weight
    roots. Pure concat/transpose expressions remain ordinary graph operations
    until the normal constant-island pass proves and freezes their recipes.
    """
    if module.execution_functions or module.metadata.get('buffer_plan'):
        raise IRSchemaError('Split GU rewrite requires a preallocation graph checkpoint.')
    old = module.node_map[node_id]
    if (old.op != 'nn.packed_dense_matmul_glu' or old.attrs.get('packed_layout') != 'k_major_n8_k16'
            or old.attrs.get('round_activation', True) or old.attrs.get('gate_clamp_max') is not None
            or old.attrs.get('up_clamp') is not None
            or any(isinstance(module.node_map[i].type, DistributedType) for i in old.inputs)):
        raise IRSchemaError('Split GU rewrite requires the original packed scalar BF16 boundary before distribution.')
    gate_id, up_id = old.inputs[1:]
    function = next((f for f in module.functions if gate_id in f.parameters and up_id in f.parameters), None)
    if function is None:
        raise IRSchemaError('Split GU readonly lifting requires the original reusable function parameters.')
    gi, ui = function.parameters.index(gate_id), function.parameters.index(up_id)
    callers = [n for n in module.nodes if n.op == 'builtin.call' and n.attrs.get('callee') == function.name]
    if not callers:
        raise IRSchemaError('Split GU has no concrete function invocations.')
    definitions = dict(module.node_map)
    fresh, replacements = [], {}

    def make(identity, op, inputs, attrs=None, metadata=None):
        values = tuple(definitions[i] for i in inputs)
        prepared = get_definition(op).prepare(values, attrs or {})
        value = Node(identity, op, tuple(inputs), prepared.result_type, attrs=prepared.attrs, effect=prepared.effect,
                     metadata=metadata or {})
        definitions[identity] = value
        fresh.append(value)
        return value

    paired_id = node_id + '.rowmajor_weight.parameter'
    paired = Node(paired_id, 'builtin.var', (), tensor_type('bfloat16', (2048, 12288)),
                  attrs={'name': 'paired_gate_up_weight_rowmajor'}, metadata={'function_parameter': function.name})
    definitions[paired_id] = paired
    for index, call in enumerate(callers):
        logical = []
        for parameter in (gi, ui):
            packed = definitions[call.inputs[parameter]]
            root = definitions.get(packed.metadata.get('packed_from', ''))
            if (root is None or root.op != 'builtin.weight' or not root.effect.is_pure
                    or root.type != tensor_type('bfloat16', (6144, 2048))):
                raise IRSchemaError('Split GU paired weight lacks original immutable logical checkpoint provenance.')
            logical.append(root.id)
        joined = make(call.id + '.gu_weight_join', 'tensors.concat', logical, {'axis': 0})
        weight = make(call.id + '.gu_weight_rowmajor', 'tensors.permute', (joined.id, ), {'axes': (1, 0)}, {
            'rdata_group': {'name': 'qwen3.layer.mlp.paired_gu.weight', 'index': index, 'count': len(callers)},
            'readonly_checkpoint_roots': tuple(logical), 'introduced_by': 'SplitKDenseGlu'
        })
        inputs = tuple(weight.id if i == gi else v for i, v in enumerate(call.inputs) if i != ui)
        metadata = {k: v for k, v in call.metadata.items() if k not in ('lifted_parameter_transforms', )}
        replacements[call.id] = replace(call, inputs=inputs, metadata=metadata)
    partial = make(
        node_id + '.split_projection', SplitKDenseGluProjection.op_name, (old.inputs[0], paired_id), metadata={
            'execution_mesh': ExecutionMesh.fixed((48, 2)).to_data(), 'kernel_program_kind': 'ordinary',
            'kernel_num_warps': 8, 'kernel_maxnreg': 64, 'introduced_by': 'SplitKDenseGlu'
        })
    prepared = SplitKDenseGluEpilogue.prepare((partial, ), {})
    replacement = replace(
        old, op=SplitKDenseGluEpilogue.op_name, inputs=(partial.id, ), attrs=prepared.attrs, type=prepared.result_type,
        effect=prepared.effect, metadata={
            'execution_mesh': ExecutionMesh.fixed((24, )).to_data(), 'kernel_program_kind': 'ordinary',
            'kernel_num_warps': 8, 'introduced_by': 'SplitKDenseGlu'
        })
    replacements[node_id] = replacement
    # Place definitions before their consumers without inventing a new
    # allocation or source mutation. The normal function graph owns order.
    fresh_by_call = {call.id: [n for n in fresh if n.id.startswith(call.id + '.gu_weight')] for call in callers}
    rewritten = []
    for node in module.nodes:
        if node.id in (gate_id, up_id):
            if node.id == gate_id:
                rewritten.append(paired)
            continue
        rewritten.extend(fresh_by_call.get(node.id, ()))
        if node.id == node_id:
            rewritten.append(partial)
        rewritten.append(replacements.get(node.id, node))
    parameters = tuple(paired_id if i == gi else value for i, value in enumerate(function.parameters) if i != ui)
    updated = replace(
        function, parameters=parameters, attrs={
            k: v
            for k, v in function.attrs.items()
            if k not in ('function_boundary_layout', 'post_auto_distributed_boundary_layout')
        })
    removed_points = {p.id for p in module.selection_points if p.owner == node_id}
    result = replace(module, nodes=tuple(rewritten), functions=tuple(updated if f.name == function.name else f
                                                                     for f in module.functions),
                     selection_points=tuple(p for p in module.selection_points if p.id not in removed_points),
                     selections=tuple(s for s in module.selections if s.point_id not in removed_points))
    from triton.flagmega.passes.functions.remove_unused import remove_unused_functions
    from triton.flagmega.passes.constants import freeze_constant_islands
    result = verify_module(freeze_constant_islands(remove_unused_functions(result)))
    # Each ordinary GU program publishes a real function result. Keep the
    # retained cooperative work in complete reusable stages, so allocation
    # and transfer lifetime planning can see the actual global boundaries.
    from triton.flagmega.passes.functions.graph import function_nodes
    from triton.flagmega.passes.functions.outline_stages import FunctionStage, outline_function_stages
    body = function_nodes(result, result.function_map[function.name])
    parameters = set(parameters)
    foreign = {n.id for f in result.functions if f.name != function.name for n in function_nodes(result, f)}
    owned = tuple(n.id for n in body if n.id not in parameters and n.id not in foreign)
    projection_index, epilogue_index = owned.index(partial.id), owned.index(node_id)
    if epilogue_index != projection_index + 1:
        raise IRSchemaError('Split GU stages require adjacent typed projection and epilogue operations.')
    stages = (FunctionStage('before_gu', owned[:projection_index]), FunctionStage('gu_projection', (partial.id, )),
              FunctionStage('gu_epilogue', (node_id, )), FunctionStage('after_gu', owned[epilogue_index + 1:]))
    outlined = outline_function_stages(result, function.name, stages)
    record = outlined.metadata['outlined_function_stages'][-1]
    bindings = {value: value for value in updated.parameters}
    wrapper_nodes = []
    for index, stage in enumerate(record['stages']):
        output_ids = stage['outputs']
        output_types = tuple(result.node_map[value].type for value in output_ids)
        result_type = output_types[0] if len(output_types) == 1 else TupleType(output_types)
        identity = f'{function.name}.gu_stage{index}'
        wrapper_nodes.append(
            Node(identity, 'builtin.call', tuple(bindings.get(value, value) for value in stage['inputs']), result_type,
                 attrs={'callee': stage['function']}, effect=callers[0].effect if index == 0 else partial.effect))
        for field, value in enumerate(output_ids):
            if len(output_ids) == 1:
                bindings[value] = identity
            else:
                projection = f'{identity}.result{field}'
                wrapper_nodes.append(
                    Node(projection, 'builtin.get_item', (identity, ), output_types[field], attrs={'index': field}))
                bindings[value] = projection
    wrapper = replace(updated, outputs=tuple(bindings.get(value, value) for value in updated.outputs))
    caller_nodes = {n.id for n in result.nodes}
    stage_parameters = {value for f in outlined.functions if f.name != result.entry for value in f.parameters}
    nodes = [n for n in outlined.nodes if n.id in caller_nodes or n.id in stage_parameters]
    # Retain the original concrete calls and share one four-stage wrapper.
    # Structured program argument binding then reuses each compiled stage
    # across all layers while preserving every actual readonly weight root.
    call_ids = {call.id for call in callers}
    existing = {n.id for n in nodes}
    prefix = [result.node_map[value] for value in updated.parameters if value not in existing]
    nodes = [*prefix, *wrapper_nodes, *(result.node_map[n.id] if n.id in call_ids else n for n in nodes)]
    return verify_module(replace(outlined, nodes=tuple(nodes), functions=(*outlined.functions, wrapper)))


__all__ = ['SplitKDenseGluDistributionPolicy', 'split_k_dense_glu_policy', 'rewrite_split_k_dense_glu']
