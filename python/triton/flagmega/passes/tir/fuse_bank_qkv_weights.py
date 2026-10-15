# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Absorb a loop body's residual offline weight tree into its parameter bank.

``CanonicalizePackedQKVWeights`` fuses three RHS weights into one and rewrites
the kernel and graph-function ABI accordingly. It clones the fusing transform
at the call site, where an unrolled call chain has ``main`` as its caller and
constant-island freezing turns the clone into readonly bytes.

A structured loop body is not ``main``: its weights come from parameter-bank
rows selected by the induction value, so the clone stays in the body and would
run once per iteration. This pass moves that remaining pure tree into the bank
fields it reads, which keeps it offline at one evaluation per row and leaves
the body with a single weight projection.

It runs after the canonicalizer, so the ABI is already correct here and only
the placement of the offline work changes.
"""

from __future__ import annotations

from dataclasses import replace

from triton.flagmega.errors import IRSchemaError, IRVerificationError
from triton.flagmega.ir import (
    DistributedType,
    IRModule,
    Node,
    get_definition,
    logical_type,
)
from triton.flagmega.ir.parameter_bank import BankField, BankFieldProgram, ParameterBankSchema
from triton.flagmega.ir.region import is_region
from triton.flagmega.passes.functions.graph import function_nodes

# Operations that constitute real offline weight work. A tree containing none
# of these is only an ABI view and is left where the canonicalizer put it.
_WEIGHT_WORK_OPS = frozenset({
    "tensors.concat",
    "tensors.pack",
    "tensors.unpack",
    "tensors.permute",
    "tensors.reshape",
    "tensors.bitcast",
    # After lower-tir a boundary layout change appears as a typed buffer
    # view, which is still offline weight work when its leaf is a bank row.
    "tir.buffer_view",
})

# Pure value-layout operations that may be evaluated per row offline.
_OFFLINE_OPS = frozenset({
    "tensors.concat",
    "tensors.pack",
    "tensors.unpack",
    "tensors.permute",
    "tensors.reshape",
    "tensors.bitcast",
    "distributed.sharded_view",
    "distributed.materialize_local_shards",
    "tir.buffer_view",
})

# Ops that read their operand's placement rather than only its logical shape.
# Their operands keep the DistributedType when a program is absorbed.
_DISTRIBUTION_AWARE_OPS = frozenset({
    "distributed.sharded_view",
    "distributed.materialize_local_shards",
})


def fuse_bank_qkv_weights(module: IRModule) -> IRModule:
    banks = {node.id: node for node in module.nodes
             if node.op in {"builtin.parameter_bank", "tir.parameter_bank"}}
    if not banks:
        return module
    region_names = {function.name for function in module.functions if is_region(function)}
    if not region_names:
        return module
    body_nodes: set[str] = set()
    for function in module.functions:
        if function.name in region_names:
            body_nodes.update(node.id for node in function_nodes(module, function))
    if not body_nodes:
        return module

    node_map = module.node_map
    users: dict[str, list[Node]] = {}
    for node in module.nodes:
        for value in node.inputs:
            users.setdefault(value, []).append(node)

    # Row projections are the storage boundary; resolve every value that
    # denotes a bank, including the loop's captured body parameter.
    bank_of_value = _bank_by_value(module, banks, region_names)
    rows = {node.id: bank_of_value[node.inputs[0]] for node in module.nodes
            if node.op in {"control.bank_get", "tir.bank_get"} and node.inputs[0] in bank_of_value}
    fields: dict[str, tuple[str, int]] = {}
    for node in module.nodes:
        if node.op == "builtin.get_item" and node.inputs and node.inputs[0] in rows:
            fields[node.id] = (rows[node.inputs[0]], int(node.attrs["index"]))
    if not fields:
        return module

    trees = _offline_trees(module, fields, users, body_nodes, node_map)
    if not trees:
        return module

    schemas: dict[str, ParameterBankSchema] = {}
    appended: dict[str, list[tuple[BankField, tuple[int, ...]]]] = {}
    redirects: dict[str, str] = {}
    removed: set[str] = set()
    for root, leaves, nodes in trees:
        bank_ids = {fields[leaf][0] for leaf in leaves}
        if len(bank_ids) != 1:
            continue
        bank_id = next(iter(bank_ids))
        schema = schemas.get(bank_id, banks[bank_id].attrs["schema"])
        indices = tuple(fields[leaf][1] for leaf in leaves)
        if len(set(indices)) != len(indices):
            continue
        field = _absorbed_field(schema, indices, leaves, nodes, root, module.stage)
        schemas[bank_id] = ParameterBankSchema(schema.name, (*schema.fields, field))
        appended.setdefault(bank_id, []).append((field, indices, root.type))
        redirects[root.id] = f"{root.id}.bank_field"
        removed.update(node.id for node in nodes)
    if not schemas:
        return module
    return _rebuild(module, banks, schemas, appended, redirects, removed, rows, fields)


def _offline_trees(module, fields, users, body_nodes, node_map):
    """Find maximal pure trees inside the body whose leaves are bank fields."""
    claimed: set[str] = set()
    results = []
    for node in module.nodes:
        if node.id not in body_nodes or node.op not in _OFFLINE_OPS or node.id in claimed:
            continue
        if node.id in fields:
            continue
        collected: list[Node] = []
        leaves: list[str] = []
        if not _collect(node, fields, body_nodes, node_map, collected, leaves, set()):
            continue
        # Only a value consumed exactly once is purely a storage decision; a
        # shared intermediate would duplicate offline work per consumer.
        if any(len(users.get(value.id, ())) != 1 for value in collected[:-1]):
            continue
        if not leaves or any(value.id in claimed for value in collected):
            continue
        parent = users.get(node.id, ())
        if len(parent) == 1 and parent[0].op in _OFFLINE_OPS and parent[0].id in body_nodes:
            # Not maximal: the parent will claim this subtree.
            continue
        # A lone boundary view is an ABI adapter, not offline weight work;
        # moving it into storage would change nothing and would hide the
        # distribution contract the callee expects.
        if not any(value.op in _WEIGHT_WORK_OPS for value in collected):
            continue
        claimed.update(value.id for value in collected)
        results.append((node, tuple(leaves), tuple(collected)))
    return tuple(results)


def _collect(node, fields, body_nodes, node_map, collected, leaves, seen) -> bool:
    if node.id in seen:
        return False
    seen.add(node.id)
    for value in node.inputs:
        source = node_map.get(value)
        if source is None:
            return False
        if value in fields:
            if value not in leaves:
                leaves.append(value)
            continue
        if source.op not in _OFFLINE_OPS or source.id not in body_nodes or not source.effect.is_pure:
            return False
        if not _collect(source, fields, body_nodes, node_map, collected, leaves, seen):
            return False
    collected.append(node)
    return True


def _absorbed_field(schema, indices, leaves, nodes, root, stage) -> BankField:
    """Build one field whose program replays the tree for a single row."""
    sources: list[str] = []
    stored: list = []
    names: dict[str, str] = {}
    for leaf, index in zip(leaves, indices, strict=True):
        field = schema.fields[index]
        if field.program is None:
            name = f"{field.name}.stored"
            names[leaf] = name
            sources.append(name)
            # Keep the distribution on a source the absorbed program still
            # distributes. The program it joins carries
            # ``distributed.materialize_local_shards``, which reads its
            # operand's placement to pick that owner's shard; reducing the
            # source to its logical type here leaves the program unable to
            # run, and anything reconstructing the layout downstream has no
            # contract left to follow.
            stored.append(field.type)
            continue
        # Compose with the layout program this field already owns.
        renamed = {
            value: f"{field.name}.{value}"
            for value in (*field.program.sources, *(value.id for value in field.program.nodes))
        }
        for value, stored_type in zip(field.program.sources, field.stored_types, strict=True):
            sources.append(renamed[value])
            stored.append(stored_type)
        names[leaf] = renamed[field.program.output]
    body: list[Node] = []
    for value in nodes:
        names.setdefault(value.id, f"{value.id}.absorbed")
    for field, index in ((schema.fields[index], index) for index in indices):
        if field.program is None:
            continue
        for value in field.program.nodes:
            body.append(replace(
                value, id=f"{field.name}.{value.id}",
                inputs=tuple(f"{field.name}.{item}" for item in value.inputs)))
    # A distribution-aware op reads its operand's placement to pick the shard
    # it materializes, so the nodes feeding one keep their DistributedType.
    # Reducing them to logical types leaves the absorbed program unable to
    # run, and anything reconstructing the owner layout downstream has no
    # contract left to follow.
    distributed_operands = {
        item
        for value in nodes
        if value.op in _DISTRIBUTION_AWARE_OPS
        for item in value.inputs
    }
    for value in nodes:
        body.append(replace(
            value, id=names[value.id],
            type=(value.type if value.id in distributed_operands
                  else logical_type(value.type)),
            inputs=tuple(names[item] for item in value.inputs)))
    program = BankFieldProgram(tuple(sources), tuple(body), names[root.id])
    produced = program.node_map[program.output].type
    if produced != logical_type(root.type):
        raise IRVerificationError(
            f"Absorbed bank program for {root.id!r} changed its result type.", stage=stage)
    transform = schema.fields[indices[0]].transform
    return BankField(f"{root.id}.bank_field", produced, transform, program, None, tuple(stored))


def _bank_parameters(module: IRModule, schemas) -> dict[str, str]:
    """Map each loop body parameter that denotes a rewritten bank to that bank."""
    result: dict[str, str] = {}
    for node in module.nodes:
        if node.op != "control.for":
            continue
        region = module.function_map.get(str(node.attrs["callee"]))
        if region is None:
            continue
        for parameter, actual in zip(region.parameters[1:], node.inputs[3:], strict=True):
            if actual in schemas:
                result[parameter] = actual
    return result


def _bank_by_value(module: IRModule, banks, region_names) -> dict[str, str]:
    result = {bank_id: bank_id for bank_id in banks}
    for node in module.nodes:
        if node.op != "control.for" or str(node.attrs["callee"]) not in region_names:
            continue
        region = module.function_map[str(node.attrs["callee"])]
        for parameter, actual in zip(region.parameters[1:], node.inputs[3:], strict=True):
            if actual in result:
                result[parameter] = result[actual]
    return result


def _rebuild(module, banks, schemas, appended, redirects, removed, rows, fields):
    # A row access infers its element type from the bank value, so capture the
    # distributed row type each bank must expose before rewriting nodes.
    bank_row_types = {}
    for value in module.nodes:
        if value.op not in {"control.bank_get", "tir.bank_get"}:
            continue
        bank = value.inputs[0] if value.inputs[0] in schemas else None
        if bank is None:
            for owner in module.nodes:
                if owner.op != "control.for":
                    continue
                region = module.function_map.get(str(owner.attrs.get("callee", "")))
                if region is None:
                    continue
                for parameter, actual in zip(region.parameters[1:], owner.inputs[3:]):
                    if parameter == value.inputs[0] and actual in schemas:
                        bank = actual
        if bank is not None:
            bank_row_types[bank] = _row_type(
                value.type, schemas[bank], appended.get(bank, ()))
    nodes: list[Node] = []
    for node in module.nodes:
        if node.id in removed:
            continue
        if node.id in schemas:
            schema = schemas[node.id]
            # A row access infers its type from the bank value, so the bank's
            # RefType must carry each field's chosen distribution; the schema
            # itself stays logical because it describes stored layout.
            nodes.append(replace(
                node, type=_bank_ref_type(schema, bank_row_types.get(node.id)),
                attrs={**dict(node.attrs), "schema": schema,
                       "rows": _extended_rows(node, appended[node.id])},
            ))
            continue
        nodes.append(node)

    # Each absorbed root becomes one projection of its bank's new field.
    projections: dict[str, list[Node]] = {}
    for bank_id, entries in appended.items():
        schema = schemas[bank_id]
        row_id = next((key for key, value in rows.items() if value == bank_id), None)
        if row_id is None:
            continue
        for offset, (field, _, root_type) in enumerate(entries):
            index = len(schema.fields) - len(entries) + offset
            # The schema field is the logical layout; the projection keeps the
            # distribution the absorbed root already had, so consumers and the
            # callee ABI see exactly the type they saw before.
            projections.setdefault(row_id, []).append(Node(
                field.name, "builtin.get_item", (row_id,), root_type, attrs={"index": index},
                metadata={"absorbed_into_parameter_bank": True}))

    # A loop captures its bank, so the body parameter carries the bank's type
    # too. Re-type it with the extended schema or the row access, which reads
    # the schema through that parameter, keeps the old field count.
    captured = _bank_parameters(module, schemas)
    result: list[Node] = []
    row_schemas: dict[str, ParameterBankSchema] = {}
    for node in nodes:
        if node.id in captured:
            # The captured bank parameter mirrors the bank value's type,
            # including the per-field distribution its rows expose.
            owner = captured[node.id]
            result.append(replace(
                node, type=_bank_ref_type(schemas[owner], bank_row_types.get(owner))))
            continue
        if node.op in {"control.bank_get", "tir.bank_get"}:
            bank_id = node.inputs[0] if node.inputs[0] in schemas else rows.get(node.id)
            schema = schemas.get(bank_id)
            if schema is not None:
                row_schemas[node.id] = schema
                result.append(replace(
                    node, type=_row_type(node.type, schema, appended.get(bank_id, ()))))
                for value in projections.get(node.id, ()):
                    result.append(value)
                continue
        inputs = tuple(redirects.get(value, value) for value in node.inputs)
        if node.op == "builtin.get_item" and node.inputs and node.inputs[0] in row_schemas:
            schema = row_schemas[node.inputs[0]]
            index = int(node.attrs["index"])
            # The schema records logical field types; a projection keeps the
            # distribution AutoDistributed chose for it, so only re-type when
            # the logical layout actually changed.
            if logical_type(node.type) == logical_type(schema.fields[index].type):
                result.append(replace(node, inputs=inputs))
                continue
            result.append(_retyped(replace(node, inputs=inputs), schema.fields[index].type))
            continue
        result.append(replace(node, inputs=inputs))
    # Absorbing a tree can orphan the field projections it used to read. A
    # per-function dump view keeps unreferenced nodes, so leaving them behind
    # produces a fragment whose row access is missing. Drop them here.
    # An absorbed node's selection point describes a choice the bank program
    # now owns, so retire it rather than leaving a dangling owner.
    points = tuple(
        point for point in module.selection_points
        if point.owner is None or point.owner not in removed
    )
    point_ids = {point.id for point in points}
    return replace(
        module,
        nodes=_without_orphans(tuple(result), module, set(fields)),
        selection_points=points,
        selections=tuple(
            record for record in module.selections if record.point_id in point_ids
        ),
    )


def _without_orphans(nodes, module, candidates):
    """Drop only the named field projections that lost their last consumer."""
    live = {value for function in module.functions for value in function.outputs}
    live.update(value for function in module.functions for value in function.parameters)
    for node in nodes:
        live.update(node.inputs)
    return tuple(node for node in nodes if node.id in live or node.id not in candidates)


def _extended_rows(bank, entries):
    """Append one key group per absorbed field to every row."""
    rows = []
    for row in bank.attrs["rows"]:
        values = list(row)
        for _, indices, _root in entries:
            keys: list = []
            for index in indices:
                value = row[index]
                keys.extend(value if isinstance(value, tuple) else (value,))
            values.append(tuple(keys) if len(keys) != 1 else keys[0])
        rows.append(tuple(values))
    return tuple(rows)


def _bank_ref_type(schema, element_type):
    """Return a bank RefType whose element keeps per-field distribution."""
    from triton.flagmega.ir import RefType

    if element_type is None:
        return schema.type
    return RefType("parameter_bank", (("element", element_type),))


def _row_type(previous, schema, appended=()):
    """Rebuild a row tuple, keeping each field's chosen distribution.

    An appended absorbed field inherits the distribution its absorbed root
    already had, so the row exposes exactly the type its consumers expect.
    """
    from triton.flagmega.ir import TupleType

    old = list(previous.fields) if isinstance(previous, TupleType) else []
    extra = {
        len(schema.fields) - len(appended) + offset: root_type
        for offset, (_, _, root_type) in enumerate(appended)
    }
    fields = []
    for index, field in enumerate(schema.fields):
        if index in extra and logical_type(extra[index]) == field.type:
            fields.append(extra[index])
            continue
        source = old[index] if index < len(old) else None
        if isinstance(source, DistributedType) and logical_type(source) == field.type:
            fields.append(source)
            continue
        fields.append(field.type)
    return TupleType(tuple(fields))


def _retyped(node, new_type):
    if isinstance(node.type, DistributedType) and not isinstance(new_type, DistributedType):
        try:
            return replace(node, type=replace(node.type, tensor=new_type))
        except IRSchemaError:
            return replace(node, type=new_type)
    return replace(node, type=new_type)


__all__ = ["fuse_bank_qkv_weights"]
