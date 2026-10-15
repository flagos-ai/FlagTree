# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Absorb per-row parameter transforms into the parameter-bank schema.

AutoPacking lifts an offline weight transform out of a reusable callee to its
call sites, where constant-island freezing turns it into readonly bytes. When
the only call site is a structured loop body, the transform's leaf is a bank
row selected by the induction value, so it is not a module constant and would
become per-iteration runtime work.

This pass moves such a chain to the bank field that feeds it: storage keeps
the checkpoint layout, the field names the transform, and the row access
exposes the physical value. The runtime loop body is then free of weight
transforms, exactly as for an unrolled call chain.
"""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import IRModule, Node
from triton.flagmega.ir.parameter_bank import BankField, BankFieldProgram, ParameterBankSchema
from triton.flagmega.ir.region import is_region
from triton.flagmega.passes.functions.graph import function_nodes


def absorb_bank_parameter_transforms(module: IRModule) -> IRModule:
    node_map = module.node_map
    banks = {node.id: node for node in module.nodes if node.op == "builtin.parameter_bank"}
    if not banks:
        return module
    region_names = {function.name for function in module.functions if is_region(function)}
    if not region_names:
        return module
    owned: set[str] = set()
    for function in module.functions:
        if function.name in region_names:
            owned.update(node.id for node in function_nodes(module, function))

    # A loop captures its bank, so the body indexes a region parameter rather
    # than the bank node. Resolve each region parameter back to the bank the
    # enclosing For passes in, and only then treat a row access as a boundary.
    bank_of_value = _bank_by_value(module, banks, region_names)
    rows = {
        node.id: bank_of_value[node.inputs[0]] for node in module.nodes
        if node.op == "control.bank_get" and node.inputs[0] in bank_of_value
    }
    fields: dict[str, str] = {}
    for node in module.nodes:
        if node.op == "builtin.get_item" and node.inputs and node.inputs[0] in rows:
            fields[node.id] = node.inputs[0]
    if not fields:
        return module

    users: dict[str, list[Node]] = {}
    for node in module.nodes:
        for value in node.inputs:
            users.setdefault(value, []).append(node)

    programs: dict[tuple[str, int], tuple[BankFieldProgram, Node, tuple[Node, ...]]] = {}
    for field_id, row_id in fields.items():
        chain = _single_use_transform_chain(field_id, users, owned)
        if not chain:
            continue
        bank = banks[rows[row_id]]
        index = int(node_map[field_id].attrs["index"])
        root = chain[-1]
        source = f"{bank.attrs['schema'].fields[index].name}.stored"
        renamed = _rename(chain, {field_id: source})
        program = BankFieldProgram(source, renamed, renamed[-1].id)
        key = (bank.id, index)
        if key in programs:
            raise IRVerificationError(
                f"Parameter bank field {index} has more than one absorbed transform.",
                stage=module.stage, node_id=root.id)
        programs[key] = (program, root, chain)
    if not programs:
        return module

    removed = {node.id for _, _, chain in programs.values() for node in chain}
    replacements = {
        root.id: field_id
        for (bank_id, index), (_, root, _) in programs.items()
        for field_id, row_id in fields.items()
        if rows[row_id] == bank_id and int(node_map[field_id].attrs["index"]) == index
    }

    nodes: list[Node] = []
    for node in module.nodes:
        if node.id in removed:
            continue
        if node.id in banks:
            schema = node.attrs["schema"]
            updated = tuple(
                _absorbed(field, programs[(node.id, index)][0])
                if (node.id, index) in programs else field
                for index, field in enumerate(schema.fields)
            )
            nodes.append(replace(
                node,
                type=ParameterBankSchema(schema.name, updated).type,
                attrs={**dict(node.attrs), "schema": ParameterBankSchema(schema.name, updated)},
            ))
            continue
        inputs = tuple(replacements.get(value, value) for value in node.inputs)
        if node.id in fields and (rows[fields[node.id]],
                                  int(node.attrs["index"])) in programs:
            bank_id = rows[fields[node.id]]
            index = int(node.attrs["index"])
            nodes.append(replace(node, inputs=inputs,
                                 type=programs[(bank_id, index)][0].node_map[
                                     programs[(bank_id, index)][0].output].type))
            continue
        nodes.append(replace(node, inputs=inputs))

    # The schema type changes, so every value that denotes the bank re-types
    # with it: the For operand, the body's captured parameter, each row
    # projection, and the field projections read off that row.
    schemas = {node.id: node.attrs["schema"] for node in nodes if node.op == "builtin.parameter_bank"}
    bank_types = {value: schemas[bank].type for value, bank in bank_of_value.items() if bank in schemas}
    typed: list[Node] = []
    row_types: dict[str, ParameterBankSchema] = {}
    for node in nodes:
        if node.id in bank_types and node.op in {"builtin.var", "builtin.parameter_bank"}:
            typed.append(node if node.op == "builtin.parameter_bank"
                         else replace(node, type=bank_types[node.id]))
            continue
        if node.op == "control.bank_get" and node.inputs[0] in bank_types:
            schema = schemas[bank_of_value[node.inputs[0]]]
            row_types[node.id] = schema
            typed.append(replace(node, type=schema.element_type))
            continue
        if node.op == "builtin.get_item" and node.inputs and node.inputs[0] in row_types:
            schema = row_types[node.inputs[0]]
            typed.append(replace(node, type=schema.fields[int(node.attrs["index"])].type))
            continue
        typed.append(node)
    return replace(module, nodes=tuple(typed))


def _bank_by_value(module: IRModule, banks, region_names) -> dict[str, str]:
    """Map every value that denotes a bank to its defining bank node id."""
    result = {bank_id: bank_id for bank_id in banks}
    for node in module.nodes:
        if node.op != "control.for" or str(node.attrs["callee"]) not in region_names:
            continue
        region = module.function_map[str(node.attrs["callee"])]
        # Region parameters are the induction value followed by the loop's
        # iter arguments and captures, in the For's own operand order.
        for parameter, actual in zip(region.parameters[1:], node.inputs[3:], strict=True):
            if actual in result:
                result[parameter] = result[actual]
    return result


def _absorbed(field: BankField, program: BankFieldProgram) -> BankField:
    if field.program is not None:
        raise IRVerificationError(f"Bank field {field.name!r} already owns a transform program.")
    return BankField(field.name, program.node_map[program.output].type, field.transform,
                     program, field.stored_type)


def _single_use_transform_chain(field_id, users, owned) -> tuple[Node, ...]:
    """Return the maximal single-use pure unary chain rooted at a row field.

    Only a linear chain is absorbed: a value consumed by real compute, or by
    more than one consumer, is not purely a storage layout decision.
    """
    chain: list[Node] = []
    current = field_id
    while True:
        consumers = users.get(current, ())
        if len(consumers) != 1:
            break
        node = consumers[0]
        if node.op not in {"tensors.pack", "tensors.unpack", "tensors.permute",
                           "tensors.reshape", "tensors.bitcast"}:
            break
        if node.id not in owned or len(node.inputs) != 1 or not node.effect.is_pure:
            break
        chain.append(node)
        current = node.id
    return tuple(chain)


def _rename(chain: tuple[Node, ...], mapping: dict[str, str]) -> tuple[Node, ...]:
    renamed: list[Node] = []
    names = dict(mapping)
    for node in chain:
        names.setdefault(node.id, f"{node.id}.absorbed")
        renamed.append(replace(
            node,
            id=names[node.id],
            inputs=tuple(names[value] for value in node.inputs),
        ))
    return tuple(renamed)


__all__ = ["absorb_bank_parameter_transforms"]
