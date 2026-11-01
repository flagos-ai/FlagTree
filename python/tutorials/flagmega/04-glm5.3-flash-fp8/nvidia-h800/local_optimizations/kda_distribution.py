# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Measured TP4 KDA ownership through reshapes, convolution and recurrence."""
from triton.flagmega import ir as fm

POLICY = "glm53-kda-chip-contiguous-heads/v2"


def _owns_heads(value):
    if isinstance(value, fm.TupleType):
        return all(_owns_heads(field) for field in value.fields)
    if (not isinstance(value, fm.DistributedType)
        or value.partial is not None or value.exclusive is not None
        or value.tensor.rank < 2 or value.tensor.shape[1].fixed_value != 64):
        return False
    head = value.axis_policies[1]
    return (
        isinstance(head, fm.SBPSplit) and tuple(head.hierarchy_axes) == (0, 2)
        and len(head.stages) == 1
        and isinstance(head.stages[0].distribution, fm.ContiguousSplit)
        and all(policy == fm.SBP.broadcast()
                for axis, policy in enumerate(value.axis_policies) if axis != 1)
    )


def selections(graph):
    """Constrain complete typed candidates; retain the ordinary coupled solve."""
    selected = {}
    for function in ("decode_linear_attention_dense", "decode_linear_attention_sparse"):
        projections = [node for node in graph.module.nodes
                       if node.op == "nn.kda_input_projection" and node.id.startswith(function + "_")]
        if len(projections) > 1:
            raise ValueError(f"{function}: expected one shared KDA input projection.")
        suffixes = ["_conv", "_recurrent.packed"]
        names = ("gate_heads", "out_gate_heads") if projections else (
            "q_proj_heads", "k_proj_heads", "v_proj_heads", "gate_heads", "out_gate_heads")
        suffixes += [f"_{name}{tail}" for name in names for tail in (".propagated.reshape", "")]
        if projections:
            name = projections[0].id
            candidates = [value for value in graph.bucket_map[name].candidates
                          if _owns_input_projection(value.return_type)]
            if len(candidates) != 1:
                raise ValueError(f"{name}: expected one complete-chip KDA projection candidate, got "
                                 f"{[value.id for value in candidates]}")
            selected[name] = candidates[0].id
        for suffix in suffixes:
            name = function + suffix
            candidates = [value for value in graph.bucket_map[name].candidates
                          if _owns_heads(value.return_type)]
            if len(candidates) != 1:
                raise ValueError(f"{name}: expected one contiguous chip/head candidate, got "
                                 f"{[value.id for value in candidates]}")
            selected[name] = candidates[0].id
    return selected


def _owns_input_projection(value):
    if not isinstance(value, fm.TupleType) or len(value.fields) != 6:
        return False
    for index, field in enumerate(value.fields):
        if not isinstance(field, fm.DistributedType) or field.partial is not None or field.exclusive is not None:
            return False
        if index < 4 and (field.tensor.shape[1].fixed_value != 64
                          or field.axis_policies[1] != fm.SBP.split_contiguous((0,))):
            return False
        if any(policy != fm.SBP.broadcast() for axis, policy in enumerate(field.axis_policies)
               if index >= 4 or axis != 1):
            return False
    return True
