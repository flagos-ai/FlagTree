# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: MIT
"""Typed inventory of representation boundaries at the two early G3 gates."""

from triton.flagmega.errors import IRVerificationError
from triton.flagmega.ir import verify_module
from triton.flagmega.ir.fusion import iter_fusions
from triton.flagmega.passes.constants import ConstantPhase, constant_phase, freeze_constant_islands
from triton.flagmega.passes.functions.graph import function_nodes


ADAPTERS = {"Pack": ("tensors.pack",), "Slice": ("tensors.slice",),
            "Cast": ("tensors.cast", "ntt.vectorized_cast"),
            "Concat": ("tensors.concat",), "Unpack": ("tensors.unpack",)}


def g3_adapter_inventory(module, *, checkpoint):
    """Inspect reachable functions/lexical regions and immutable recipe proofs.

    Constant partitioning runs on an immutable copy, without reading weights
    or evaluating runtime values. Only nodes actually inside those recipes
    are classified offline; labels and parameter names carry no authority.
    """
    verify_module(module)
    pending, visited, owners = [module.entry], set(), {}
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for node in function_nodes(module, module.function_map[name]):
            owners.setdefault(node.id, []).append(name)
            if node.op in {"builtin.call", "tir.call", "control.for"}:
                callee = str(node.attrs.get("callee", ""))
                if callee in module.function_map:
                    pending.append(callee)
    proof = freeze_constant_islands(module) if constant_phase(module) == ConstantPhase.OPEN else module
    recipes = {node.id: recipe for recipe in proof.constant_recipes for node in recipe.nodes}
    rows, fused = [], []
    for node in module.nodes:
        if node.id not in owners:
            continue
        family = next((name for name, ops in ADAPTERS.items() if node.op in ops), None)
        if family is not None:
            recipe = recipes.get(node.id)
            rows.append({"id": node.id, "op": node.op, "family": family,
                         "functions": owners[node.id], "type": node.type.to_data(),
                         "producers": list(node.inputs),
                         "consumers": [user.id for user in module.nodes if node.id in user.inputs],
                         "offline_recipe": None if recipe is None else
                         {"id": recipe.id, "fingerprint": recipe.fingerprint, "outputs": list(recipe.outputs)}})
        bodies = tuple(iter_fusions(node.attrs))
        if bodies:
            # Reuse the owning emitter's capability proof. A Fusion object is
            # not an exemption for unsupported expressions or hidden copies.
            from triton.flagmega.codegen.triton.fusion import can_fuse
            if not can_fuse(node, module):
                raise IRVerificationError("G3: unsupported PreOps/PostOps owner.",
                                          stage=module.stage, node_id=node.id)
            for body in bodies:
                fused.append({"owner": node.id, "fingerprint": body.fingerprint,
                              "conversions": [{"id": value.id, "op": value.op,
                                               "type": value.type.to_data(),
                                               "rounding_contract": value.attrs.get("rounding_contract")}
                                              for value in body.nodes if value.op in ADAPTERS["Cast"]]})
    return {"schema": "flagmega.typed-g3-inventory/v1", "checkpoint": checkpoint,
            "stage": module.stage, "semantic_hash": module.semantic_hash,
            "entry": module.entry, "reachable_functions": sorted(visited),
            "runtime_counts": {name: sum(row["family"] == name and row["offline_recipe"] is None
                                         for row in rows) for name in ADAPTERS},
            "offline_counts": {name: sum(row["family"] == name and row["offline_recipe"] is not None
                                         for row in rows) for name in ADAPTERS},
            "scalar_casts": sum(row["op"] == "tensors.cast" and row["offline_recipe"] is None for row in rows),
            "numeric_vector_casts": sum(row["op"] == "ntt.vectorized_cast" and row["offline_recipe"] is None
                                        for row in rows),
            "adapters": rows, "supported_fusions": fused}


def check_g3_adapters(module, *, checkpoint):
    inventory = g3_adapter_inventory(module, checkpoint=checkpoint)
    runtime = [row for row in inventory["adapters"] if row["offline_recipe"] is None]
    if runtime:
        row = runtime[0]
        raise IRVerificationError(
            f"G3 at {checkpoint}: runtime {row['op']} remains; "
            f"producers={row['producers']}, consumers={row['consumers']}; counts={inventory['runtime_counts']}.",
            stage=module.stage, node_id=row["id"])
    return inventory


__all__ = ["ADAPTERS", "g3_adapter_inventory", "check_g3_adapters"]
