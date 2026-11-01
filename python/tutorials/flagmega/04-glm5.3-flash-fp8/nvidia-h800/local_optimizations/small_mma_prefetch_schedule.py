# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Select the ordered MMA prefetch implementation for measured small rows."""

from dataclasses import replace
from math import prod

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.selection import apply_plan, override_plan
from triton.flagmega.targets.nvidia.dense_matmul import packed_mma_prefetch_implementations

IMPLEMENTATION = "tir.dense_matmul.packed_k_major_mma_prefetch"
POLICY = "glm53-small-bf16-ordered-mma-prefetch/v1"


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
    (implementation,) = packed_mma_prefetch_implementations()
    existing = base.implementation(IMPLEMENTATION)
    if existing is not None:
        if existing != implementation:
            raise IRVerificationError("The MMA prefetch implementation is registered with a different contract.")
        return base
    return replace(base, implementations=(*base.implementations, implementation))


def select(module):
    choices = []
    for point in module.selection_points:
        if point.kind != "tir" or point.owner not in module.node_map:
            continue
        node = module.node_map[point.owner]
        record = module.selection_map.get(point.id)
        if (
            node.op != "ntt.packed_matmul"
            or not node.inputs
            or record is None
            or record.candidate_id != "tir.dense_matmul.packed_k_major_mma"
        ):
            continue
        columns = _row_shape(node.type)
        reduction = _row_shape(module.node_map[node.inputs[0]].type)
        if (
            columns is not None
            and 1 <= columns <= 16
            and reduction is not None
            and reduction >= 1024
            and reduction % 16 == 0
        ):
            if IMPLEMENTATION not in {candidate.id for candidate in point.candidates}:
                raise IRVerificationError(f"{point.id}: the ordered MMA prefetch candidate is unavailable.")
            choices.append((point.id, IMPLEMENTATION))
    return apply_plan(
        module,
        override_plan(
            module,
            choices,
            policy=POLICY,
            rationale="Measured small complete-K BF16 row; preserve K32 MMA arithmetic with Shared prefetch.",
        ),
    )
