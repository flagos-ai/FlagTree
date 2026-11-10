# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Form a single projection's complementary, head-interleaved result fields."""

from dataclasses import replace
from collections import defaultdict
from triton.flagmega.ir import verify_module, TensorType, DType
from triton.flagmega.ir.ops.tensors.slice import Slice
from triton.flagmega.rules.neutral._utility import make_node


def form_interleaved_projection(module):
    """Replace only proven complete complementary field views, once per producer.

    Any observation of the original full projection/factored view rejects
    formation, so expensive projections are never cloned for mixed users.
    No effectful operation or state update is moved or merged.
    """
    users = defaultdict(list)
    for node in module.nodes:
        for source in node.inputs:
            users[source].append(node)
    exported = {n for f in module.functions for n in (*f.parameters, *f.outputs)}
    replacements, removed = {}, set()
    for projection in module.nodes:
        if (projection.op != "math.block_scaled_matmul" or projection.attrs.get("weight_encoding") is None
                or projection.attrs.get("output_data_type") not in (None, "bfloat16")
                or not isinstance(projection.type, TensorType) or projection.type.dtype != DType.BFLOAT16
                or projection.id in exported or len(users[projection.id]) != 1):
            continue
        factored = users[projection.id][0]
        if factored.op != "tensors.reshape" or factored.id in exported or factored.type.rank != 4:
            continue
        shape = factored.type.shape
        if any(not d.is_fixed for d in shape) or shape[2].fixed_value != 2:
            continue
        _, heads, _, width = (d.fixed_value for d in shape)
        if projection.type.shape != (shape[0], shape[1] * shape[2] * shape[3]):
            continue
        selected, intermediates, seen = [], [], set()
        valid = True
        for selection in users[factored.id]:
            if selection.op != "tensors.slice" or selection.id in exported:
                valid = False; break
            ranges = Slice.ranges(factored.type, selection.attrs)
            changed = [(a, r) for a, r in ranges if r != range(shape[a].fixed_value)]
            if len(changed) != 1 or changed[0][0] != 2 or len(changed[0][1]) != 1 or changed[0][1].step != 1:
                valid = False; break
            field = changed[0][1].start
            if field not in (0, 1) or not users[selection.id]:
                valid = False; break
            seen.add(field)
            for view in users[selection.id]:
                if view.op != "tensors.reshape" or view.type.shape != (shape[0], shape[1], shape[3]):
                    valid = False; break
                selected.append((view, field))
            intermediates.append(selection.id)
            if not valid: break
        if not valid or seen != {0, 1}:
            continue
        inputs = tuple(module.node_map[n] for n in projection.inputs)
        formed = make_node("nn.interleaved_projection", projection.id, inputs,
                           {k: projection.attrs[k] for k in ("weight_block_n", "weight_block_k", "weight_encoding")}
                           | {"feature_shape": (heads, width)},
                           {**projection.metadata, "formed_by": "FormInterleavedProjection"})
        replacements[projection.id] = formed
        for view, field in selected:
            replacement = make_node("builtin.get_item", view.id, (formed,), {"index": field},
                                    {**view.metadata, "formed_by": "FormInterleavedProjection"})
            assert replacement.type == view.type
            replacements[view.id] = replacement
        removed.update((factored.id, *intermediates))
    if not replacements:
        return module
    return verify_module(replace(module, nodes=tuple(replacements.get(n.id, n) for n in module.nodes if n.id not in removed)))


__all__ = ["form_interleaved_projection"]
