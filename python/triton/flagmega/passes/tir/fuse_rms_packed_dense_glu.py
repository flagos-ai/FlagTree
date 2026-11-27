# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""A closed, typed distributed RMS chain into complete-row GLU staging."""
from collections import Counter
from dataclasses import dataclass, replace
from math import prod

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir import DistributedType, Node, SBP, TensorLayout, VectorType, verify_module
from triton.flagmega.ir.distributed_inference import tensor_of
from triton.flagmega.ir.distributed_type import ReduceOp, SBPSplit, local_tensor_type
from triton.flagmega.ir.op_fusion import has_ops
from triton.flagmega.ir.ops.core import get_definition
from triton.flagmega.ir.ops.ntt.rms_packed_dense_matmul_glu import RmsPackedDenseMatMulGlu
from triton.flagmega.ir.ops.tensors.bitcast import Bitcast


@dataclass(frozen=True)
class CompleteRmsGluMatch:
    glu: str
    raw: str
    scale: str
    bias: str
    norm: str
    stats: str
    tree_group_size: int
    removed: tuple[str, ...]


def _scalar_geometry(value_type):
    tensor = tensor_of(value_type)
    if tensor.layout != TensorLayout() or not tensor.shape:
        return None
    dtype = tensor.dtype
    scalar = dtype.elem_type if isinstance(dtype, VectorType) else dtype
    lanes = dtype.lane_count if isinstance(dtype, VectorType) else 1
    return scalar, (*tensor.shape[:-1], tensor.shape[-1] * lanes)


def _view(node, nodes):
    if len(node.inputs) != 1 or has_ops(node.attrs) or not node.effect.is_pure:
        return False
    source = nodes[node.inputs[0]]
    if _scalar_geometry(node.type) != _scalar_geometry(source.type):
        return False
    if node.op in {'distributed.sharded_view', 'distributed.boxing'}:
        return tensor_of(node.type) == tensor_of(source.type)
    if node.op == 'tensors.bitcast':
        return (Bitcast.zero_copy_input_index((source,), node.attrs, node.type) == 0
                and Bitcast.infer_type((source,), node.attrs) == node.type)
    return False


def find_complete_rms_glu_matches(module):
    nodes = module.node_map
    uses = Counter(key for node in module.nodes for key in node.inputs)
    # Returns/parameter escapes are uses too, including reusable callees.
    uses.update(key for function in module.functions for key in (*function.parameters, *function.outputs))
    matches = []
    for glu in module.nodes:
        if glu.op != 'nn.packed_dense_matmul_glu' or has_ops(glu.attrs):
            continue
        value, removed = nodes[glu.inputs[0]], []
        while _view(value, nodes) and uses[value.id] == 1:
            removed.append(value.id)
            value = nodes[value.inputs[0]]
        norm = value
        if (norm.op != 'nn.norm_apply' or len(norm.inputs) != 4 or uses[norm.id] != 1
                or has_ops(norm.attrs) or norm.attrs['use_mean']):
            continue
        raw, materialized, scale, bias = (nodes[key] for key in norm.inputs)
        raw_type = raw.type
        if (not isinstance(raw_type, DistributedType) or raw_type.partial is not None
                or raw_type.exclusive is not None or len(raw_type.axis_policies) != 2
                or not isinstance(raw_type.axis_policies[1], SBPSplit)
                or not raw_type.axis_policies[1].is_contiguous):
            continue
        raw_geometry = _scalar_geometry(raw_type)
        if raw_geometry is None or raw_geometry[0].value != 'float32':
            continue
        if (int(norm.attrs['axis']) % 2 != 1
                or norm.attrs.get('input_dtype', 'float32') != 'float32'
                or norm.attrs.get('output_dtype') != 'bfloat16'):
            continue
        if (materialized.op != 'distributed.boxing' or uses[materialized.id] != 1
                or len(materialized.inputs) != 1 or has_ops(materialized.attrs)):
            continue
        stats = nodes[materialized.inputs[0]]
        if (stats.op != 'nn.norm_stats' or stats.inputs != (raw.id,) or uses[stats.id] != 1
                or has_ops(stats.attrs) or stats.attrs['use_mean']
                or int(stats.attrs['axis']) % 2 != 1
                or any(stats.attrs.get(name, default) != default for name, default in (
                    ('reduction_order', 'unspecified'), ('reduction_group_size', 1),
                    ('reduction_block_size', None), ('reduction_tree_group_size', None),
                    ('reduction_group_seed_index', 0)))):
            continue
        partial = getattr(stats.type, 'partial', None)
        if (partial is None or partial.reduce_op is not ReduceOp.SUM
                or set(partial.axes) != set(raw_type.axis_policies[1].hierarchy_axes)
                or materialized.type != replace(stats.type, partial=None)):
            continue
        local = local_tensor_type(raw_type)
        if any(not d.is_fixed for d in local.shape):
            continue
        group = local.shape[-1].fixed_value * (local.dtype.lane_count if isinstance(local.dtype, VectorType) else 1)
        owners = prod(raw_type.placement.hierarchy[axis] for axis in partial.axes)
        global_k = raw_geometry[1][-1].fixed_value
        if (group <= 0 or group & (group - 1) or owners & (owners - 1)
                or global_k != group * owners):
            continue
        # Two actual local trees, not an arbitrary reduction hidden by boxing.
        matches.append(CompleteRmsGluMatch(glu.id, raw.id, scale.id, bias.id, norm.id,
                                         stats.id, group, (*removed, norm.id, materialized.id, stats.id)))
    return tuple(matches)


def fuse_rms_packed_dense_glu(module, *, partial_group_size):
    """Run after vector-contract lowering, before TIR/physical planning."""
    if type(partial_group_size) is not int or partial_group_size <= 0 or partial_group_size & (partial_group_size - 1):
        raise IRSchemaError('RMS/GLU fusion requires an explicit positive power-of-two partial group contract.')
    verify_module(module)
    if module.dialect == 'semantic_tir' or any(key in module.metadata for key in ('buffer_plan', 'function_memory_placement')):
        raise IRVerificationError('RMS/GLU fusion must resume before TIR and PlanFunctionMemory.', stage=module.stage)
    if any('bufferization.memory_space' in node.metadata for node in module.nodes):
        raise IRVerificationError('RMS/GLU fusion cannot retain late memory placement metadata.', stage=module.stage)
    matches = find_complete_rms_glu_matches(module)
    if not matches:
        return module
    nodes = module.node_map
    replacements, insertions, removed = {}, {}, set()
    occupied = set(nodes)

    def create(op, arguments, attrs, stem):
        name, ordinal = stem, 0
        while name in occupied:
            ordinal += 1
            name = f'{stem}.{ordinal}'
        prepared = get_definition(op).prepare(arguments, attrs)
        value = Node(name, op, tuple(arg.id for arg in arguments), prepared.result_type,
                     effect=prepared.effect, attrs=prepared.attrs,
                     metadata={'introduced_by': 'FuseRmsPackedDenseGlu'})
        occupied.add(name)
        return value

    for match in matches:
        glu, norm = nodes[match.glu], nodes[match.norm]
        added, operands = [], []
        for role, key in (('raw', match.raw), ('scale', match.scale), ('bias', match.bias)):
            value = nodes[key]
            if not isinstance(value.type, DistributedType) or value.type.partial is not None or value.type.exclusive is not None:
                break
            broad = replace(value.type, axis_policies=(SBP.broadcast(),) * tensor_of(value.type).rank)
            if broad != value.type:
                value = create('distributed.sharded_view', (value,), {'new_type': broad}, f'{glu.id}.rms_{role}.broadcast')
                added.append(value)
            dtype = tensor_of(value.type).dtype
            if isinstance(dtype, VectorType):
                value = create('tensors.bitcast', (value,), {'dtype': dtype.elem_type}, f'{glu.id}.rms_{role}.scalar')
                added.append(value)
            operands.append(value)
        if len(operands) != 3:
            continue
        operands.extend(nodes[key] for key in glu.inputs[1:])
        attrs = {**glu.attrs, 'epsilon': norm.attrs['epsilon'],
                 'norm_tree_group_size': match.tree_group_size,
                 'norm_partial_group_size': partial_group_size,
                 'norm_math_mode': norm.attrs.get('math_mode', 'fast'),
                 'round_before_scale': norm.attrs.get('round_before_scale', False)}
        try:
            prepared = RmsPackedDenseMatMulGlu.prepare(operands, attrs)
        except IRSchemaError:
            continue
        if prepared.result_type != glu.type:
            continue
        replacements[glu.id] = replace(glu, op=RmsPackedDenseMatMulGlu.op_name,
            inputs=tuple(v.id for v in operands), attrs=prepared.attrs, effect=prepared.effect,
            metadata={**glu.metadata, 'introduced_by': 'FuseRmsPackedDenseGlu'})
        insertions[glu.id] = added
        removed.update(match.removed)
    if not replacements:
        return module
    invalid = {p.id for p in module.selection_points if p.owner in removed or p.owner in replacements}
    rewritten = []
    for node in module.nodes:
        if node.id in removed:
            continue
        rewritten.extend(insertions.get(node.id, ()))
        rewritten.append(replacements.get(node.id, node))
    return verify_module(replace(module, nodes=tuple(rewritten),
        selection_points=tuple(p for p in module.selection_points if p.id not in invalid),
        selections=tuple(s for s in module.selections if s.point_id not in invalid)))


__all__ = ['CompleteRmsGluMatch', 'find_complete_rms_glu_matches', 'fuse_rms_packed_dense_glu']
