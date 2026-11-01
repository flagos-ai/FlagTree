# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Explicit GEMV schedule for small, complete-K BF16 decode projections.

The existing implementation has a different FP32 reduction tree from MMA.
Official full-model token acceptance is required before enabling this policy.
"""

from dataclasses import replace
from math import prod

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.selection import apply_plan, override_plan

IMPLEMENTATION = 'tir.dense_matmul.packed_k_major_gemv_tn16_bk128'
POLICY = 'glm53-small-bf16-complete-k-gemv/v1'


def _row_shape(value_type):
    distributed = value_type if isinstance(value_type, fm.DistributedType) else None
    if distributed is not None and (distributed.partial is not None or distributed.exclusive is not None):
        return None
    kind = fm.local_tensor_type(distributed) if distributed is not None else value_type
    if not isinstance(kind, fm.TensorType) or len(kind.shape) != 2 or kind.shape[0].fixed_value != 1:
        return None
    dtype = kind.dtype
    scalar = dtype.elem_type if isinstance(dtype, fm.VectorType) else dtype
    if scalar is not fm.DType.BFLOAT16:
        return None
    columns = kind.shape[1].fixed_value
    return None if columns is None else columns * (prod(dtype.lanes) if isinstance(dtype, fm.VectorType) else 1)


def implementation_model(base):
    """Configure the existing catalog entry before candidates are proposed."""
    implementation = base.implementation(IMPLEMENTATION)
    assert implementation is not None
    assert implementation.variant == 'packed_k_major_gemv'
    assert not implementation.shared_workspaces and implementation.transfer_pipeline is None
    return replace(base, implementations=tuple(
        replace(value, parameters={**value.parameters, 'compute_num_warps': 8})
        if value.id == IMPLEMENTATION else value for value in base.implementations))


def select(module):
    """Choose declared alternatives before lowering, bufferization and sync."""
    choices = []
    for point in module.selection_points:
        if point.kind != 'tir' or point.owner not in module.node_map:
            continue
        node = module.node_map[point.owner]
        record = module.selection_map.get(point.id)
        if (node.op != 'ntt.packed_matmul' or not node.inputs or record is None
                or record.candidate_id != 'tir.dense_matmul.packed_k_major_mma'):
            continue
        columns = _row_shape(node.type)
        reduction = _row_shape(module.node_map[node.inputs[0]].type)
        if (columns is not None and 1 <= columns <= 16 and reduction is not None
                and reduction >= 1024 and reduction % 128 == 0):
            if IMPLEMENTATION not in {candidate.id for candidate in point.candidates}:
                raise IRVerificationError(f'{point.id}: the small BF16 GEMV candidate is unavailable.')
            choices.append((point.id, IMPLEMENTATION))
    return apply_plan(module, override_plan(module, choices, policy=POLICY,
        rationale='Single-row complete-K BF16 projection with at most 16 physical outputs per owner.'))
