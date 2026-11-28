# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Prove which ParameterBank indexing ABI a row can use."""

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir.op_fusion import has_ops
from triton.flagmega.passes.functions.graph import function_nodes


def bank_row_index(module, function, row, bank_id):
    """Return a fixed row, or None for a proved loop-to-function row ABI.

    The existing dynamic ABI supplies the counted-loop induction value to a
    reusable callee's pooled tensor argument. A direct leaf or an unrelated
    scalar index has no such address carrier, and must not read row zero.
    """
    index = module.node_map[row.inputs[1]]
    if index.op in {"builtin.scalar_const", "tir.scalar_const"}:
        value = index.attrs["value"]
        extent = len(module.node_map[bank_id].attrs["rows"])
        if type(value) is not int or not 0 <= value < extent:
            raise IRVerificationError("Bank row index is outside its stored domain.", node_id=row.id)
        return value

    loops = [node for node in module.nodes if node.op == "control.for" and node.attrs.get("callee") == function.name]
    if not loops or not function.parameters or row.inputs[1] != function.parameters[0]:
        raise IRVerificationError(
            "Dynamic bank row requires the counted-loop induction argument and a pooled callee boundary.",
            node_id=row.id)
    nodes = function_nodes(module, function)
    users = {}
    for node in nodes:
        for operand in node.inputs:
            users.setdefault(operand, []).append(node)
    pending = [row.id]
    visited = set()
    calls = 0
    while pending:
        value = pending.pop()
        if value in visited:
            continue
        visited.add(value)
        for user in users.get(value, ()):
            if (user.op in {"builtin.get_item", "builtin.tuple", "distributed.sharded_view", "tir.buffer_view"}
                    and user.effect.is_pure and not has_ops(user.attrs)):
                pending.append(user.id)
            elif user.op == "tir.call" and user.attrs.get("callee") in module.function_map:
                calls += 1
            else:
                raise IRVerificationError("Dynamic bank row reaches a leaf without a pooled callee index binding.",
                                          node_id=user.id)
    if not calls:
        raise IRVerificationError("Dynamic bank row has no pooled callee index binding.", node_id=row.id)
    return None
