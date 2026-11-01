# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Select ordered register loads and disjoint fragments of complete N8 tiles."""

from dataclasses import replace
from math import prod

from triton.flagmega import ir as fm
from triton.flagmega.errors import IRVerificationError
from triton.flagmega.selection import apply_plan, override_plan


ORIGINAL = "tir.dense_matmul.packed_k_major_threaded_fma"
IMPLEMENTATION = "tir.dense_matmul.packed_k_major_threaded_fma_register8_n8_s2"
POLICY = "glm53-ordered-fma-register8-cooperation/v1"


def implementation_model(base):
    original = base.implementation(ORIGINAL)
    expected = {"tile_n": 8, "block_k": 1024, "dot_lanes": 128, "values_per_lane": 8}
    if original is None or any(original.parameters.get(k) != v for k, v in expected.items()):
        raise IRVerificationError("Ordered register loads require the complete N8/K1024 FMA contract.")
    implementation = replace(original, id=IMPLEMENTATION, parameters={
        **original.parameters, "weight_load_layout": "register8", "scalar_store_subtile_n": 2})
    existing = base.implementation(IMPLEMENTATION)
    if existing is not None:
        if existing != implementation:
            raise IRVerificationError("Conflicting ordered register-load implementation.")
        return base
    return replace(base, implementations=(*base.implementations, implementation))


def _can_cooperate(module, node):
    result = node.type
    if not isinstance(result, fm.DistributedType):
        return False
    local = fm.local_tensor_type(result)
    if (not isinstance(local, fm.TensorType) or local.rank != 2
            or local.shape[0].fixed_value != 1
            or not isinstance(local.dtype, fm.VectorType)
            or local.dtype.elem_type is not fm.DType.FLOAT32 or local.dtype.lane_count != 8):
        return False
    columns = local.shape[1].fixed_value
    if columns is None or columns * 8 <= 8:
        return False
    source = module.node_map[node.inputs[0]].type
    if not isinstance(source, fm.DistributedType):
        return False
    source_local = fm.local_tensor_type(source)
    k = source_local.shape[-1].fixed_value
    if k is None or k < 1024 or k % 1024:
        return False
    used = set()
    for value in (node, *(module.node_map[i] for i in node.inputs)):
        kind = value.type
        if isinstance(kind, fm.TensorType):
            return False  # This policy needs explicit distributed ownership.
        if not isinstance(kind, fm.DistributedType):
            continue
        if kind.placement != result.placement or kind.partial is not None or kind.exclusive is not None:
            return False
        for policy in kind.axis_policies:
            used.update(getattr(policy, "hierarchy_axes", ()))
    placement = result.placement
    count = prod(size for axis, size in enumerate(placement.hierarchy)
                 if placement.is_physical_block_axis(axis) and axis not in used)
    # The renderer additionally proves the actual shared/private storage and
    # address invariance; a logical replica count alone never authorizes writes.
    return count >= 4 and count % 4 == 0


def select(module):
    choices = []
    for point in module.selection_points:
        record = module.selection_map.get(point.id)
        if point.kind != "tir" or record is None or record.candidate_id != ORIGINAL:
            continue
        node = module.node_map.get(point.owner)
        if node is None or node.op != "ntt.packed_matmul" or not _can_cooperate(module, node):
            continue
        if IMPLEMENTATION not in {candidate.id for candidate in point.candidates}:
            raise IRVerificationError(f"{point.id}: ordered register-load candidate is unavailable.")
        choices.append((point.id, IMPLEMENTATION))
    if not choices:
        return module
    return apply_plan(module, override_plan(module, choices, policy=POLICY,
        rationale="Ordered BF16 FMA; complete logical N8 tiles, four disjoint scalar2 fragments, "
                  "and block replicas with all input/result owners fixed. Physical proof is mandatory at codegen."))
